"""
SwitcherEnv: Gym-like environment wrapper for CAPSwitcher PPO training.

At each step the switcher commits to one mode for ``selection_interval``
simulation steps, then re-observes.

Observation space
-----------------
(N, 512) float32 — per-robot decoder-output embeddings (``attn_out``), kept
unpooled.  The Deep Sets switcher head learns its own permutation-invariant
aggregation over robots, so the environment does **not** pool here (raw pooling
of navigation embeddings is lossy — see deep_sets_head.py).

Action space
------------
Discrete(2):  0 = coarse,  1 = precise.

Reward
------
Sum over sub-steps of:
    mean(per-robot simulation reward from phase 6)  +  mode_cost

where mode_cost = −0.2 (coarse) or −1.0 (precise) per sub-step.

Episode termination
-------------------
- any robot collision
- all robots reached their goals
- step count reaches max_steps
- any robot outside world bounds
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_nav.models.MARL.capswitcher.rl.reward import MODE_COSTS


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
        selection_interval: Number of sim steps committed to one mode per
                            switcher decision. Default 5.
        max_steps:          Maximum sim steps per episode. Default 300.
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
        max_steps: int = 300,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.sim = sim
        self.backbone = backbone
        self.coarse = coarse_steering
        self.selection_interval = selection_interval
        self.max_steps = max_steps
        self.device = device

        self._step_count: int = 0
        self._robot_state: np.ndarray | None = None
        self._obstacle_states: np.ndarray | None = None
        self._poses: list | None = None
        self._last_action: list | None = None
        self._goal_positions: list | None = None

        # ---- Forward-pass cache (Plan A: avoid redundant GAT forwards) ----
        # Each backbone forward returns both the precise actions and the
        # pre-decoder embedding for the *current* robot state.  We cache both
        # so that (a) the observation does not trigger a second forward on a
        # state we already ran, and (b) precise mode reuses the actions that
        # were produced alongside the observation that led to this decision.
        # ``_cache_valid`` is False after a coarse sub-step (which needs no
        # forward) and is lazily refreshed when the observation is requested.
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
        self._poses = poses
        self._last_action = a
        self._goal_positions = goal_positions
        self._obstacle_states = obstacle_states

        robot_state, _ = self.backbone.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )
        self._robot_state = np.array(robot_state, dtype=np.float32)

        # Prime the cache for the initial state.
        self._cache_valid = False
        self._refresh_cache()

        return self._get_obs()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute ``selection_interval`` sim steps with the chosen mode.

        Args:
            action: 0 = coarse steering, 1 = precise (frozen GAT actor).

        Returns:
            obs:    (N, 512) next observation.
            reward: Accumulated reward over all sub-steps.
            done:   Episode termination flag.
            info:   Dict with diagnostic keys:
                    ``collision``, ``all_reached``, ``timeout``, ``oob``,
                    ``mode``, ``steps_taken``.
        """
        accumulated_reward = 0.0
        mode_cost = MODE_COSTS[action]
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

        # ---- Pre-build the sub-step frames for this decision ----------
        # Coarse: one coarse control of a randomly chosen group expands into a
        # sequence of rotation sub-steps (A-matrix steering, fully realised)
        # followed by a single forward move sub-step.  The whole control is
        # computed once from the decision-time state.  Precise: re-derive the
        # actor action each sub-step (state-dependent), as before.
        if action == 0:
            group = int(self._coarse_rng.choice(self.coarse.selectable_groups()))
            rotation_frames, move_frame = self.coarse.compute_actions(
                self._robot_state, group
            )
            coarse_frames = rotation_frames + [move_frame]
            n_substeps = len(coarse_frames)
            info["group"] = group
        else:
            coarse_frames = None
            n_substeps = self.selection_interval

        for sub in range(n_substeps):
            # ---- Choose sim-input actions based on mode ---------------
            if action == 0:
                # Coarse: pre-built rotation/move frame, already sim-input.
                # Needs no backbone forward.
                sim_actions = coarse_frames[sub]
            else:
                # Precise: reuse the actions produced by the cached forward for
                # the current state (computed either by the previous decision's
                # observation or the previous sub-step), so we never re-run the
                # GAT on a state we already evaluated.  Convert actor-space to
                # sim-input: lin_vel = (raw[0]+1)/4 ∈ [0, 0.5], ang_vel = raw[1].
                if not self._cache_valid:
                    self._refresh_cache()
                raw_actions = self._cached_raw_actions
                sim_actions = [
                    [
                        (float(raw_actions[i, 0]) + 1.0) / 4.0,
                        float(raw_actions[i, 1]),
                    ]
                    for i in range(self.sim.num_robots)
                ]

            # ---- Step the simulator ------------------------------------
            (
                poses, distance, cos, sin, collision, goal, a, sim_rewards,
                positions, goal_positions, obstacle_states,
            ) = self.sim.step(sim_actions, None, None)

            self._step_count += 1
            info["steps_taken"] = sub + 1

            # ---- Accumulate reward -------------------------------------
            mean_sim_reward = float(np.mean(sim_rewards))
            accumulated_reward += mean_sim_reward + mode_cost

            # ---- Update cached state ----------------------------------
            robot_state, _ = self.backbone.prepare_state(
                poses, distance, cos, sin, collision, a, goal_positions
            )
            self._robot_state = np.array(robot_state, dtype=np.float32)
            self._obstacle_states = obstacle_states
            self._poses = poses
            self._last_action = a
            self._goal_positions = goal_positions

            # ---- Refresh / invalidate the forward-pass cache ----------
            # Precise mode needs the next-state forward to produce the next
            # sub-step's actions, so run it now (it doubles as the observation
            # forward at the end).  Coarse mode needs no forward for actions, so
            # defer it: mark the cache stale and let _get_obs run a single
            # forward on the final state only.
            if action == 1:
                self._refresh_cache()
            else:
                self._cache_valid = False

            # ---- Termination checks -----------------------------------
            if any(collision):
                info["collision"] = True
                done = True
            if all(goal):
                info["all_reached"] = True
                done = True
            if self._step_count >= self.max_steps:
                info["timeout"] = True
                done = True
            if _outside_bounds(poses, self.sim):
                info["oob"] = True
                done = True

            if done:
                break

        return self._get_obs(), accumulated_reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_cache(self) -> None:
        """
        Run one frozen GAT forward on the current state and cache the result.

        Populates ``_cached_raw_actions`` (N, 2 actor-space) and ``_cached_h``
        ((N, 512) CPU tensor), and marks the cache valid.  This is the only
        place the backbone is invoked, so each distinct robot state is run at
        most once per sub-step.
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

        Reuses the cached forward when valid (precise sub-steps and reset
        already populate it); otherwise runs a single forward on the final
        state (the coarse-mode path).  No pooling is applied — the Deep Sets
        switcher head aggregates over robots itself.

        Returns:
            (N, 512) numpy float32 array.
        """
        if not self._cache_valid:
            self._refresh_cache()
        # _cached_h: (N, 512) CPU tensor → (N, 512) numpy
        return self._cached_h.numpy().astype(np.float32)
