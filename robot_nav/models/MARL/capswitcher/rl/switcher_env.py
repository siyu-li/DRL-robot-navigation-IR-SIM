"""
SwitcherEnv: Gym-like environment wrapper for CAPSwitcher DQN training.

At each step the switcher commits to one mode for the duration of that mode's
control, then re-observes:

  * Coarse  — one coarse control of a randomly chosen group expands into a
    sequence of rotation sub-steps followed by translation sub-steps (members
    advance by the full ``move_distance``).  ~10–14 sim sub-steps.
  * Precise — robots are resolved **one at a time**: each robot runs its frozen
    individual GAT policy for ``selection_interval`` sub-steps while the others
    hold still.  ``N × selection_interval`` sim sub-steps total (30 for 6 robots
    at 5 sub-steps each).

Observation space
-----------------
(N, 512) float32 — per-robot decoder-output embeddings (``attn_out``), kept
unpooled.  The Deep Sets switcher head learns its own permutation-invariant
aggregation over robots, so the environment does **not** pool here.

Action space
------------
Discrete(2):  0 = coarse,  1 = precise.

Reward
------
Computed **once per decision** by :class:`SwitcherReward` (see reward.py), not
summed over sub-steps:

    r = k_p · Σ_i Δdistance_i  −  Σ_i (cl_pen_i + obs_pen_i)  +  step_penalty
        (+ terminal collision / all-reached bonuses)

Progress is the summed goal-distance reduction over the whole decision; the
clearance/obstacle penalties are evaluated once at decision end.  This is
agnostic to how many sub-steps a mode consumes.

Episode termination
-------------------
- any robot collision
- all robots reached their goals
- decision count reaches ``max_decisions`` (budget is counted in *decisions*,
  not sim sub-steps, so coarse and precise episodes get the same number of
  switcher choices)
- any robot outside world bounds
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_nav.models.MARL.capswitcher.rl.reward import (
    StepPenaltyReward,
    SwitcherReward,
)


def _outside_bounds(poses: list, sim) -> bool:
    """Return True if any robot pose is outside the world boundary."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


class SwitcherEnv:
    """
    Environment wrapper for training the CAPSwitcher binary switcher.

    Args:
        sim:                MARL_SIM_OBSTACLE instance (already initialised).
        backbone:           GATBackbone (frozen) — provides embeddings and
                            precise actions.
        coarse_steering:    CoarseSteering instance.
        selection_interval: Number of sim sub-steps each robot is driven by its
                            individual policy during a precise decision. Default 5.
        max_decisions:      Maximum number of switcher decisions per episode
                            (episode budget counted in decisions, not sub-steps).
                            Default 60.
        reward_fn:          Optional reward callable (:class:`SwitcherReward` or
                            :class:`StepPenaltyReward`). Defaults to a
                            :class:`SwitcherReward` with the agreed constants.
        device:             Torch device used by the backbone.
    """

    NUM_ACTIONS: int = 2   # {0: coarse, 1: precise}
    EMBED_DIM: int = 512   # per-robot decoder-output embedding width

    def __init__(
        self,
        sim,
        backbone,
        coarse_steering,
        selection_interval: int = 5,
        max_decisions: int = 60,
        reward_fn: SwitcherReward | StepPenaltyReward | None = None,
        device: torch.device = torch.device("cpu"),
        terminate_on_oob: bool = True,
    ) -> None:
        self.sim = sim
        self.backbone = backbone
        self.coarse = coarse_steering
        self.selection_interval = selection_interval
        self.max_decisions = max_decisions
        self.reward_fn = reward_fn if reward_fn is not None else SwitcherReward()
        self.device = device
        self.terminate_on_oob = terminate_on_oob

        self._step_count: int = 0          # sim sub-steps (diagnostic only)
        self._decision_count: int = 0      # switcher decisions (episode budget)
        self._robot_state: np.ndarray | None = None
        self._obstacle_states: np.ndarray | None = None
        self._poses: list | None = None
        self._last_action: list | None = None
        self._goal_positions: list | None = None
        self._last_distances: np.ndarray | None = None  # per-robot goal distances

        # ---- Forward-pass cache (avoid redundant GAT forwards) ----
        # Holds the precise actions and the embedding for the *current* robot
        # state.  ``_cache_valid`` is False after the state changes (every
        # sub-step) and is refreshed lazily when actions or the observation are
        # requested, so each distinct state is forwarded at most once.
        self._cached_h: torch.Tensor | None = None        # (N, 512) CPU tensor
        self._cached_raw_actions: np.ndarray | None = None  # (N, 2) actor space
        self._cache_valid: bool = False

        # RNG for the random coarse move-group choice (no switcher selects the
        # group yet, so it is sampled uniformly from the selectable groups).
        self._coarse_rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, random_obstacles: bool = True) -> np.ndarray:
        """
        Reset the simulation and return the initial observation.

        Returns:
            obs: (N, 512) per-robot decoder-output embeddings (numpy float32).
        """
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, obstacle_states,
        ) = self.sim.reset(random_obstacles=random_obstacles)

        self._step_count = 0
        self._decision_count = 0
        self._poses = poses
        self._last_action = a
        self._goal_positions = goal_positions
        self._obstacle_states = obstacle_states
        self._last_distances = np.asarray(distance, dtype=np.float64)

        robot_state, _ = self.backbone.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )
        self._robot_state = np.array(robot_state, dtype=np.float32)

        # Prime the cache for the initial state.
        self._cache_valid = False
        self._refresh_cache()

        return self._get_obs()

    def step(
        self,
        action: int,
        group: int | None = None,
        frames: list | None = None,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute one switcher decision with the chosen mode.

        Args:
            action: 0 = coarse steering, 1 = precise (sequential GAT actor).
            group:  Optional 1-based coarse group to drive (action 0 only).
                    Defaults to the legacy uniform-random choice.
            frames: Optional pre-built coarse frames to execute verbatim
                    (action 0 only).  Used by the safety shield so the plan that
                    runs is exactly the plan that was vetted; takes precedence
                    over ``group`` for the rollout (``group`` is still recorded).

        Returns:
            obs:    (N, 512) next observation.
            reward: Scalar decision-level reward (see :class:`SwitcherReward`).
            done:   Episode termination flag.
            info:   Dict with diagnostic keys:
                    ``collision``, ``all_reached``, ``timeout``, ``oob``,
                    ``mode``, ``group``, ``steps_taken``.
        """
        self._decision_count += 1
        d_start = self._last_distances.copy()

        done = False
        info: dict[str, Any] = {
            "collision":   False,
            "all_reached": False,
            "timeout":     False,
            "oob":         False,
            "mode":        action,
            "group":       None,
            "steps_taken": 0,
        }

        if action == 0:
            done = self._run_coarse(info, group=group, frames=frames)
        else:
            done = self._run_precise(info)

        # ---- Decision-budget timeout (counted in decisions, not sub-steps) ---
        if not done and self._decision_count >= self.max_decisions:
            info["timeout"] = True
            done = True

        # ---- Decision-level reward ------------------------------------------
        d_end = self._last_distances
        cl_pen, obs_pen = self.sim.proximity_penalties()
        reward = self.reward_fn(
            d_start, d_end, action, cl_pen, obs_pen,
            info["collision"], info["all_reached"],
        )

        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Mode rollouts
    # ------------------------------------------------------------------

    def _run_coarse(
        self,
        info: dict[str, Any],
        group: int | None = None,
        frames: list | None = None,
    ) -> bool:
        """
        Execute one coarse control.

        The group is, in order of precedence: the pre-vetted ``frames`` (run
        verbatim, e.g. from the safety shield), an explicit ``group`` (frames
        recomputed for it), or the legacy uniform-random choice.  The whole
        control (rotation + translation sub-steps) is applied as pre-built
        frames; no backbone forward is needed during the sub-steps.

        Returns:
            done: True if a terminal condition fired during the control.
        """
        if frames is not None:
            info["group"] = group
            coarse_frames = frames
        else:
            if group is None:
                group = int(self._coarse_rng.choice(self.coarse.selectable_groups()))
            info["group"] = group
            rotation_frames, translation_frames = self.coarse.compute_actions(
                self._robot_state, group
            )
            coarse_frames = rotation_frames + translation_frames

        for sub, frame in enumerate(coarse_frames):
            done = self._apply_substep(frame, info)
            info["steps_taken"] = sub + 1
            # Coarse needs no forward for its actions; defer the embedding
            # forward to _get_obs on the final state only.
            self._cache_valid = False
            if done:
                return True
        return False

    def _run_precise(self, info: dict[str, Any]) -> bool:
        """
        Resolve robots one at a time with their frozen individual GAT policy.

        Each robot is driven for ``selection_interval`` sub-steps while the
        others hold still; its action is recomputed every sub-step (state
        dependent) from a full-graph backbone forward.

        Returns:
            done: True if a terminal condition fired during the rollout.
        """
        n = self.sim.num_robots
        steps = 0
        for r in range(n):
            for _ in range(self.selection_interval):
                # Fresh actions for the current state (lazy: one forward/state).
                if not self._cache_valid:
                    self._refresh_cache()
                raw = self._cached_raw_actions
                # Only robot r moves; convert actor-space to sim-input
                # (lin_vel = (raw[0]+1)/4 ∈ [0, 0.5], ang_vel = raw[1]).
                sim_actions = [
                    [(float(raw[i, 0]) + 1.0) / 4.0, float(raw[i, 1])]
                    if i == r else [0.0, 0.0]
                    for i in range(n)
                ]
                done = self._apply_substep(sim_actions, info)
                steps += 1
                info["steps_taken"] = steps
                # State changed → cached actions/embedding are stale.
                self._cache_valid = False
                if done:
                    return True
        return False

    def _apply_substep(
        self, sim_actions: list, info: dict[str, Any]
    ) -> bool:
        """
        Step the simulator once with ``sim_actions``, update cached state and
        termination flags.

        Returns:
            done: True if a terminal condition fired on this sub-step.
        """
        (
            poses, distance, cos, sin, collision, goal, a, sim_rewards,
            positions, goal_positions, obstacle_states,
        ) = self.sim.step(sim_actions, None, None)

        self._step_count += 1

        # ---- Update cached state ----------------------------------------
        robot_state, _ = self.backbone.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )
        self._robot_state = np.array(robot_state, dtype=np.float32)
        self._obstacle_states = obstacle_states
        self._poses = poses
        self._last_action = a
        self._goal_positions = goal_positions
        self._last_distances = np.asarray(distance, dtype=np.float64)

        # ---- Termination checks -----------------------------------------
        done = False
        if any(collision):
            info["collision"] = True
            done = True
        if all(goal):
            info["all_reached"] = True
            done = True
        if _outside_bounds(poses, self.sim):
            info["oob"] = True
            if self.terminate_on_oob:
                done = True
        return done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_cache(self) -> None:
        """
        Run one frozen GAT forward on the current state and cache the result.

        Populates ``_cached_raw_actions`` (N, 2 actor-space) and ``_cached_h``
        ((N, 512) CPU tensor), and marks the cache valid.  This is the only
        place the backbone is invoked, so each distinct robot state is run at
        most once.
        """
        raw_actions, h = self.backbone.get_embedding_and_actions(
            self._robot_state, self._obstacle_states
        )
        self._cached_raw_actions = raw_actions
        self._cached_h = h
        self._cache_valid = True

    def _get_obs(self) -> np.ndarray:
        """
        Return the per-robot decoder-output embeddings for the current state.

        Reuses the cached forward when valid; otherwise runs a single forward on
        the final state.  No pooling is applied — the Deep Sets switcher head
        aggregates over robots itself.

        Returns:
            (N, 512) numpy float32 array.
        """
        if not self._cache_valid:
            self._refresh_cache()
        return self._cached_h.numpy().astype(np.float32)
