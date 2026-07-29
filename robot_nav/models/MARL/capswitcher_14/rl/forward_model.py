"""
Analytic forward model with **single-edge transitions** (no ``sim.step`` in the
tree) — the 14-robot counterpart of ``capswitcher.rl.forward_model``.

The decisive difference from the 6-robot model: there is no ``coarse_moves``
that vets *every* group at a node.  The lazy search materialises one edge at a
time, so the model's unit of work is :meth:`coarse_move` — vet **one** group
(frames + swept clearance + progress) and, if the shield admits it, build its
deterministic next state.  Verifying a coarse action and computing its
transition are the same computation, so verify-on-descent costs nothing beyond
the transition the search must pay anyway.

Budget accounting
-----------------
``n_transitions = n_coarse_vets + n_precise_expansions`` counts every expensive
model operation: each single-group vet (including ones the shield refutes) and
each precise-all rollout.  This is the search budget unit — a per-node "expand
everything" scheme would be charged its true cost of ~23 transitions here.

Determinism
-----------
``CoarseSteering14`` never drops actuation columns, so a coarse control is a
pure function of (state, group) — the seed threading of the 6-robot model
(``_state_group_seed``) is gone entirely.

Reused from ``capswitcher`` (N-generic): shield sweep geometry
(``swept_positions`` / ``min_member_clearance`` / ``predicted_progress`` /
``CoarseCandidate`` / ``ShieldGeometry``) and ``PathCostReward``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE, PathCostReward
from robot_nav.models.MARL.capswitcher.rl.shield import (
    CoarseCandidate,
    ShieldGeometry,
    min_member_clearance,
    predicted_progress,
    swept_positions,
)


def _angle_wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into (−π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class ModelState:
    """Dynamic planner state for one lookahead node."""

    poses: np.ndarray         # (N, 3) [x, y, theta]
    last_actions: np.ndarray  # (N, 2) last applied sim-input [lin, ang]


@dataclass
class CoarseMove:
    """One vetted coarse edge: shield candidate + (if safe) the state it leads to."""

    candidate: CoarseCandidate       # group, frames, clearance, progress, safe
    next_state: ModelState | None    # None iff the shield refuted the move


class ForwardModel14:
    """
    Deterministic analytic model over robot poses for the lazy tree search.

    Args:
        backbone:           Frozen ``GATBackbone`` (precise actions + state prep).
        coarse:             ``CoarseSteering14`` primitive (shared with the env).
        goals:              (N, 2) fixed goal positions for the episode.
        obstacle_states:    (M, 4) obstacle observations for the GAT forward
                            (frozen within a decision).
        geom:               ``ShieldGeometry`` for the safety vet and collision
                            prediction.
        step_time:          Simulator sub-step duration (s).
        selection_interval: Sub-steps each robot is driven per precise decision.
        lin_max:            Max linear velocity (m/s) — used by the cost-to-go α.
        d_safe:             Clearance margin (m) a coarse move must keep to be safe.
        goal_threshold:     Per-robot goal-arrival radius (m) for ``all_reached``.
        reward_fn:          ``PathCostReward`` supplying per-decision motion cost.
        leaf_value:         Optional learned leaf evaluator ``(model, ms) -> float``.
    """

    def __init__(
        self,
        backbone,
        coarse,
        goals: np.ndarray,
        obstacle_states: np.ndarray,
        geom: ShieldGeometry,
        step_time: float,
        selection_interval: int = 5,
        lin_max: float = 0.5,
        d_safe: float = 0.3,
        goal_threshold: float = 0.3,
        reward_fn: PathCostReward | None = None,
        leaf_value=None,
    ) -> None:
        self.backbone = backbone
        self.coarse = coarse
        self.goals = np.asarray(goals, dtype=np.float64)          # (N, 2)
        self.obstacle_states = np.asarray(obstacle_states)         # (M, 4)
        self.geom = geom
        self.step_time = float(step_time)
        self.selection_interval = int(selection_interval)
        self.lin_max = float(lin_max)
        self.d_safe = float(d_safe)
        self.goal_threshold = float(goal_threshold)
        self.reward_fn = reward_fn if reward_fn is not None else PathCostReward()
        self.leaf_value = leaf_value

        self.N = self.goals.shape[0]

        # Budget counters: every single-group vet and every precise rollout.
        self.n_coarse_vets = 0
        self.n_precise_expansions = 0

    @property
    def n_transitions(self) -> int:
        """Total expensive model operations — the search budget unit."""
        return self.n_coarse_vets + self.n_precise_expansions

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    @staticmethod
    def state_from_robot_state(robot_state: np.ndarray) -> ModelState:
        """
        Recover a :class:`ModelState` from an 11-col ``robot_state`` row block
        (inverse of the GAT node-feature scalings — see the 6-robot model).
        """
        s = np.asarray(robot_state, dtype=np.float64)
        px, py = s[:, 0], s[:, 1]
        theta = np.arctan2(s[:, 3], s[:, 2])
        poses = np.stack([px, py, theta], axis=1)                  # (N, 3)
        lin = s[:, 7] / 2.0
        ang = s[:, 8] * 2.0 - 1.0
        last_actions = np.stack([lin, ang], axis=1)                # (N, 2)
        return ModelState(poses=poses, last_actions=last_actions)

    def robot_state(self, ms: ModelState) -> np.ndarray:
        """
        Build the (N, 11) GAT state for ``ms`` — mirrors ``sim.step`` geometry
        then ``backbone.prepare_state`` so layout/scaling match training.
        """
        pos = ms.poses[:, :2]
        theta = ms.poses[:, 2]
        goal_vecs = self.goals - pos                               # (N, 2)
        dist = np.linalg.norm(goal_vecs, axis=1)                   # (N,)
        dist_safe = np.where(dist > 1e-10, dist, 1e-10)
        h = np.stack([np.cos(theta), np.sin(theta)], axis=1)       # (N, 2) unit
        g = goal_vecs / dist_safe[:, None]
        cos_e = np.sum(h * g, axis=1)                              # (N,)
        sin_e = h[:, 0] * g[:, 1] - h[:, 1] * g[:, 0]             # (N,)

        states, _ = self.backbone.prepare_state(
            ms.poses.tolist(),
            dist.tolist(),
            cos_e.tolist(),
            sin_e.tolist(),
            [False] * self.N,
            ms.last_actions.tolist(),
            self.goals.tolist(),
        )
        return np.asarray(states, dtype=np.float32)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def goal_distances(self, ms: ModelState) -> np.ndarray:
        """Per-robot Euclidean distance to goal."""
        return np.linalg.norm(self.goals - ms.poses[:, :2], axis=1)

    def all_reached(self, ms: ModelState) -> bool:
        """True iff every robot is within ``goal_threshold`` of its goal."""
        return bool(np.all(self.goal_distances(ms) <= self.goal_threshold))

    def collision_pred(self, ms: ModelState) -> bool:
        """
        Geometric collision prediction for a node's state (robot–robot and
        robot–obstacle, same geometry as the shield).
        """
        pos = ms.poses[:, :2]
        rho = self.geom.rho
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        if d.min() < 2.0 * rho:
            return True
        if self.geom.obstacle_xy.shape[0] > 0:
            od = np.linalg.norm(
                pos[:, None, :] - self.geom.obstacle_xy[None, :, :], axis=2
            )
            clr = od - self.geom.obstacle_r[None, :] - rho
            if clr.min() < 0.0:
                return True
        return False

    # ------------------------------------------------------------------
    # Transitions (single-edge — the budgeted operations)
    # ------------------------------------------------------------------

    def coarse_move(self, ms: ModelState, group: int) -> CoarseMove:
        """
        Vet **one** coarse group at ``ms`` and, if the shield admits it, build
        its deterministic next state.

        This is a budgeted operation (``n_coarse_vets``), charged whether or not
        the vet succeeds — a refuted vet consumed the same sweep computation.
        A refuted move returns ``next_state=None``; the caller prunes the edge
        from the node's legal action set (no cost, no penalty — the action is
        simply not available there).
        """
        self.n_coarse_vets += 1
        rs = self.robot_state(ms)
        rot, trans = self.coarse.compute_actions(rs, group)
        samples = swept_positions(rs, rot, trans, self.step_time)  # (T+1, N, 2)
        members = self.coarse.members_of(group)
        clearance = min_member_clearance(samples, members, self.geom)
        progress = predicted_progress(rs, members, samples[-1])
        safe = clearance >= self.d_safe
        candidate = CoarseCandidate(
            group=group,
            frames=rot + trans,
            clearance=clearance,
            progress=progress,
            safe=safe,
        )
        next_state = (
            self._coarse_next_state(ms, rot, trans, samples[-1]) if safe else None
        )
        return CoarseMove(candidate=candidate, next_state=next_state)

    def _coarse_next_state(
        self,
        ms: ModelState,
        rotation_frames: list,
        translation_frames: list,
        end_positions: np.ndarray,
    ) -> ModelState:
        """
        Next state after a coarse control: swept end positions + post-rotation
        headings; ``last_actions`` = the final executed frame.
        """
        theta = ms.poses[:, 2].copy()
        for frame in rotation_frames:
            theta = theta + np.asarray(frame, dtype=np.float64)[:, 1] * self.step_time
        theta = _angle_wrap(theta)

        poses = np.empty_like(ms.poses)
        poses[:, :2] = end_positions
        poses[:, 2] = theta

        all_frames = rotation_frames + translation_frames
        last = (
            np.asarray(all_frames[-1], dtype=np.float64)
            if all_frames
            else np.zeros((self.N, 2))
        )
        return ModelState(poses=poses, last_actions=last)

    def precise_next(self, ms: ModelState) -> ModelState:
        """
        Next state after one precise-all decision: resolve robots one at a time,
        each driven by its frozen GAT action for ``selection_interval`` sub-steps
        while the others hold still, integrating unicycle dynamics.
        """
        self.n_precise_expansions += 1
        poses = ms.poses.copy()
        last = ms.last_actions.copy()
        for r in range(self.N):
            for _ in range(self.selection_interval):
                rs = self.robot_state(ModelState(poses=poses, last_actions=last))
                raw, _ = self.backbone.get_embedding_and_actions(
                    rs, self.obstacle_states
                )
                lin = (float(raw[r, 0]) + 1.0) / 4.0   # actor→sim: [0, 0.5]
                ang = float(raw[r, 1])
                sim_actions = np.zeros((self.N, 2), dtype=np.float64)
                sim_actions[r] = (lin, ang)
                poses = self._integrate(poses, sim_actions)
                last = sim_actions
        return ModelState(poses=poses, last_actions=last)

    def _integrate(self, poses: np.ndarray, sim_actions: np.ndarray) -> np.ndarray:
        """One unicycle sub-step for all robots (forward Euler, pre-update heading)."""
        dt = self.step_time
        v = sim_actions[:, 0]
        w = sim_actions[:, 1]
        theta = poses[:, 2]
        out = poses.copy()
        out[:, 0] = poses[:, 0] + v * np.cos(theta) * dt
        out[:, 1] = poses[:, 1] + v * np.sin(theta) * dt
        out[:, 2] = _angle_wrap(theta + w * dt)
        return out

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def step_cost(self, action: int, group: int | None = None) -> float:
        """
        Per-decision motion cost = ``−PathCostReward`` for a non-terminal
        decision.  Coarse ⇒ ``n_members(group) · move_distance`` (3/4/7 pricing
        is automatic); precise ⇒ flat ``precise_cost``.  Known without vetting —
        lazy branch stubs carry exact step costs from creation.
        """
        if action == COARSE:
            n_moved = int(self.coarse.members_of(group).size)
            coarse_cost = n_moved * self.coarse.move_distance
            reward = self.reward_fn(
                None, None, COARSE, None, None, False, False,
                coarse_cost=coarse_cost, oob=False,
            )
        else:
            reward = self.reward_fn(
                None, None, PRECISE, None, None, False, False,
            )
        return -float(reward)

    def cost_to_go(self, ms: ModelState, alpha: float | None = None) -> float:
        """
        Leaf cost-to-go: the configured ``leaf_value`` if set, else the analytic
        precise-completion heuristic ``α · Σ_i ‖p_i − goal_i‖``.
        """
        if self.leaf_value is not None:
            return float(self.leaf_value(self, ms))
        if alpha is None:
            alpha = self.reward_fn.precise_cost / (self.lin_max * self.step_time)
        return float(alpha) * float(np.sum(self.goal_distances(ms)))


# robot_state goal columns (same layout as the 6-robot system).
_GX, _GY = 9, 10


def build_forward_model(
    backbone,
    coarse,
    sim,
    robot_state: np.ndarray,
    *,
    d_safe: float,
    selection_interval: int,
    goal_threshold: float,
    reward_fn: PathCostReward,
    default_rho: float,
    leaf_value=None,
) -> ForwardModel14:
    """
    Rebuild the deterministic forward model from the live sim + root state —
    the one place a switcher turns (backbone, coarse, sim, root robot_state)
    into a :class:`ForwardModel14`.
    """
    s = np.asarray(robot_state, dtype=np.float64)
    goals = s[:, [_GX, _GY]]
    return ForwardModel14(
        backbone=backbone,
        coarse=coarse,
        goals=goals,
        obstacle_states=sim.get_obstacle_states(),
        geom=ShieldGeometry.from_sim(sim, default_rho=default_rho),
        step_time=coarse.step_time,
        selection_interval=selection_interval,
        lin_max=coarse.lin_max,
        d_safe=d_safe,
        goal_threshold=goal_threshold,
        reward_fn=reward_fn,
        leaf_value=leaf_value,
    )
