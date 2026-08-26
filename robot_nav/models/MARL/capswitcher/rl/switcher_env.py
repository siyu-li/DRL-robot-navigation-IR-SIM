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

Cost (reward = −cost)
---------------------
Priced **once per decision** from the :class:`SwitcherCost` table (see
``rl/cost.py``); there are no terminal bonuses — collision / all-reached /
out-of-bounds are ``done`` flags in ``info``, not reward terms:

    coarse  : cost.coarse_cost(group)          (configured per-group constant)
    precise : precise_unit × (precise sub-steps actually executed)

One robot moves per precise sub-step, so the executed sub-step count *is*
robots × sub-steps; robots already within ``goal_threshold`` are skipped by
the rollout and never charged.

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

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost


def seed_episode(env: "SwitcherEnv", seed: int) -> None:
    """
    Seed every generator one episode of ``env`` draws from, before ``reset()``.

    Three independent sources feed an episode, and missing any one of them
    makes it unreplayable:

    * ``random`` — robot start poses and the inactive-robot draw
      (``MARL_SIM_OBSTACLE.reset`` uses ``random.uniform`` / ``random.sample``);
    * ``numpy.random`` — the legacy global.  Used by policy code, and — the
      part that is easy to miss — by **irsim itself**: on the pinned ir-sim 2.x
      the **obstacles** (``env.random_obstacle_position`` → ``np.random.uniform``)
      and the **goals** (``set_random_goal`` → ``random_point_range`` →
      ``np.random.uniform``) both draw from this one global.  Without it,
      re-running "episode k" gives the same robot start poses on a *different*
      obstacle layout with *different* goals — i.e. not the same episode at all;
    * ``env._coarse_rng`` — the uniform group draw of the coarse-only baseline.

    ir-sim 3.x moved its draws off the legacy global onto a private
    module-level Generator seeded via ``irsim.util.random.set_seed``.  That
    module does not exist on 2.x, so it is seeded best-effort: present on 3.x,
    skipped on 2.x where ``np.random.seed`` above already covers those draws.

    Seeding makes an episode index a reproducible handle: the same ``--seed``
    and episode number replay the same world in ``render_gaz14``.
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)

    try:  # ir-sim >= 3 only; on 2.x np.random.seed above already covers it
        from irsim.util.random import set_seed as _irsim_set_seed
    except ImportError:
        pass
    else:
        _irsim_set_seed(seed)

    env._coarse_rng = np.random.default_rng(seed)


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
        cost:               :class:`SwitcherCost` decision-pricing table (load
                            with ``SwitcherCost.from_yaml``).
        goal_threshold:     Distance (m) below which a robot counts as reached —
                            the precise rollout skips (and never charges) such
                            robots. Default 0.3.
        device:             Torch device used by the backbone.
        coupling:           Optional ``PreciseCoupling`` — the physics fix
                            (redesign §2): a driven robot's rotation is realised
                            through the actuation matrix, so every bystander
                            receives its coupled side-rotation command (but
                            never translates).  ``None`` keeps the legacy
                            independent-rotation behaviour.
        precise_groups:     Optional list of member-index lists (redesign §3).
                            When set, ``step(1, pgroup=k)`` drives group ``k``'s
                            unreached members simultaneously instead of the
                            legacy all-robots sequential resolution (which
                            remains available via ``pgroup=None``).
    """

    NUM_ACTIONS: int = 2   # {0: coarse, 1: precise}
    EMBED_DIM: int = 512   # per-robot decoder-output embedding width

    def __init__(
        self,
        sim,
        backbone,
        coarse_steering,
        selection_interval: int = 5,
        max_decisions: int = 80,
        cost: SwitcherCost | None = None,
        goal_threshold: float = 0.3,
        device: torch.device = torch.device("cpu"),
        terminate_on_oob: bool = True,
        coupling=None,
        precise_groups: list | None = None,
    ) -> None:
        if cost is None:
            raise ValueError(
                "SwitcherEnv requires a SwitcherCost (load the system's "
                "cost YAML with SwitcherCost.from_yaml)"
            )
        self.sim = sim
        self.backbone = backbone
        self.coarse = coarse_steering
        self.selection_interval = selection_interval
        self.max_decisions = max_decisions
        self.cost = cost
        self.goal_threshold = float(goal_threshold)
        self.device = device
        self.terminate_on_oob = terminate_on_oob
        self.coupling = coupling
        self.precise_groups = (
            None if precise_groups is None
            else [np.asarray(g, dtype=int) for g in precise_groups]
        )

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
        pgroup: int | None = None,
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
            pgroup: Optional precise-group id (action 1 only; requires
                    ``precise_groups``): drive that group's unreached members
                    simultaneously instead of the all-robots sequential
                    resolution.

        Returns:
            obs:    (N, 512) next observation.
            reward: ``−path_cost`` of the decision (no terminal bonuses).
            done:   Episode termination flag.
            info:   Dict with diagnostic keys:
                    ``collision``, ``all_reached``, ``timeout``, ``oob``,
                    ``mode``, ``group``, ``steps_taken``, ``robots_moved``,
                    ``path_cost``, plus the collision-attribution keys
                    ``collision_robots`` (indices flagged by the sim) and
                    ``active_robot`` (the robot the precise rollout was
                    driving when the sub-step fired; ``None`` for coarse).
        """
        self._decision_count += 1

        done = False
        info: dict[str, Any] = {
            "collision":    False,
            "all_reached":  False,
            "timeout":      False,
            "oob":          False,
            "mode":         action,
            "group":        None,
            "steps_taken":  0,
            "robots_moved": 0,
            "path_cost":    0.0,
            # Collision attribution: which robots the sim flagged, and which
            # robot precise mode was driving at the time (precise moves exactly
            # one robot per sub-step, so this pins the blame).
            "collision_robots": [],
            "active_robot":     None,
            # Precise pricing unit: sub-steps summed over driven robots.  For
            # the legacy sequential rollout this equals ``steps_taken`` (one
            # robot per sub-step); for group mode it is |driven| per sub-step.
            "robot_substeps":   0,
            "pgroup":           None,
        }

        if action == 0:
            done = self._run_coarse(info, group=group, frames=frames)
        else:
            done = self._run_precise(info, pgroup=pgroup)

        # ---- Decision-budget timeout (counted in decisions, not sub-steps) ---
        if not done and self._decision_count >= self.max_decisions:
            info["timeout"] = True
            done = True

        # ---- Decision cost (reward = −cost, no terminal bonuses) ------------
        # Coarse: the chosen group's configured constant from the cost table.
        # Precise: one robot moves per executed sub-step, so the sub-step count
        # is robots × sub-steps; reached robots were skipped and cost nothing.
        if action == 0:
            if info["group"] is not None:
                info["path_cost"] = float(self.cost.coarse_cost(info["group"]))
                info["robots_moved"] = int(
                    self.coarse.members_of(info["group"]).size
                )
        else:
            info["path_cost"] = float(
                self.cost.precise_substep_cost(info["robot_substeps"])
            )

        reward = -info["path_cost"]
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

    def _run_precise(
        self, info: dict[str, Any], pgroup: int | None = None
    ) -> bool:
        """
        Execute one precise decision.

        ``pgroup=None`` (legacy / config A): resolve robots one at a time with
        their frozen individual GAT policy — each unreached robot is driven for
        ``selection_interval`` sub-steps while the others hold position.

        ``pgroup=k`` (configs B/C; requires ``precise_groups``): drive group
        ``k``'s unreached members **simultaneously** for ``selection_interval``
        sub-steps.

        In both modes, with a configured ``coupling`` the driven set's angular
        commands are realised through the actuation matrix (redesign §2) —
        bystanders receive their coupled side-rotation but never a linear
        command.  Robots already within ``goal_threshold`` are skipped and
        never charged.

        Returns:
            done: True if a terminal condition fired during the rollout.
        """
        if pgroup is not None:
            return self._run_precise_group(info, pgroup)
        n = self.sim.num_robots
        steps = 0
        for r in range(n):
            if float(self._last_distances[r]) <= self.goal_threshold:
                continue
            info["robots_moved"] += 1
            info["active_robot"] = r
            for _ in range(self.selection_interval):
                # Fresh actions for the current state (lazy: one forward/state).
                if not self._cache_valid:
                    self._refresh_cache()
                raw = self._cached_raw_actions
                # Only robot r advances; convert actor-space to sim-input
                # (lin_vel = (raw[0]+1)/4 ∈ [0, 0.5], ang_vel = raw[1]).
                if self.coupling is not None:
                    w = self.coupling.coupled_ang([r], [float(raw[r, 1])])
                    sim_actions = [
                        [(float(raw[i, 0]) + 1.0) / 4.0 if i == r else 0.0,
                         float(w[i])]
                        for i in range(n)
                    ]
                else:
                    sim_actions = [
                        [(float(raw[i, 0]) + 1.0) / 4.0, float(raw[i, 1])]
                        if i == r else [0.0, 0.0]
                        for i in range(n)
                    ]
                done = self._apply_substep(sim_actions, info)
                steps += 1
                info["steps_taken"] = steps
                info["robot_substeps"] = steps  # one driven robot per sub-step
                # State changed → cached actions/embedding are stale.
                self._cache_valid = False
                if done:
                    return True
        return False

    def _run_precise_group(self, info: dict[str, Any], pgroup: int) -> bool:
        """Drive one precise group's unreached members simultaneously."""
        if self.precise_groups is None:
            raise ValueError("step(pgroup=...) requires precise_groups")
        members = self.precise_groups[pgroup]
        driven = [
            int(r) for r in members
            if float(self._last_distances[r]) > self.goal_threshold
        ]
        info["pgroup"] = int(pgroup)
        info["robots_moved"] = len(driven)
        if not driven:
            return False                       # no-op edge (search filters these)
        n = self.sim.num_robots
        driven_set = set(driven)
        for _ in range(self.selection_interval):
            if not self._cache_valid:
                self._refresh_cache()
            raw = self._cached_raw_actions
            if self.coupling is not None:
                w = self.coupling.coupled_ang(
                    driven, [float(raw[r, 1]) for r in driven]
                )
            else:
                w = np.zeros(n)
                for r in driven:
                    w[r] = float(raw[r, 1])
            sim_actions = [
                [(float(raw[i, 0]) + 1.0) / 4.0 if i in driven_set else 0.0,
                 float(w[i])]
                for i in range(n)
            ]
            done = self._apply_substep(sim_actions, info)
            info["steps_taken"] += 1
            info["robot_substeps"] += len(driven)
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
            info["collision_robots"] = [
                i for i, c in enumerate(collision) if c
            ]
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
