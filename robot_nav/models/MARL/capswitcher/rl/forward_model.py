"""
Analytic forward model shared by every planner (no ``sim.step`` in the tree).

The simulator cannot be branched (irsim's internal collision tree / ``_world`` /
``step_status`` are opaque and un-cloneable), so every lookahead — the minimin
MPC and the budgeted tree searches in ``rl/search/`` — runs on this standalone
pose-model and only the *chosen first action* is executed on the real
``SwitcherEnv.step``.  Two facts make the model exact/faithful enough to rank actions:

  * within one decision **obstacles are frozen**, so obstacle geometry is a constant;
  * the coarse primitive is deterministic given its frames, and the frozen GAT is a
    pure function of state.

Transitions
-----------
* **Coarse** — exact and sim-free: reuse the shield's
  :func:`swept_positions` to reconstruct the members' straight-line advance along
  their post-rotation heading.  The column drop is seeded by ``(poses, group)`` so
  ``coarse(x, g)`` is a single deterministic transition.
* **Precise (all robots)** — faithful: mirror
  :meth:`SwitcherEnv._run_precise` — resolve robots one at a time, each driven by its
  frozen GAT action for ``selection_interval`` sub-steps (others hold still), with a
  unicycle integrator standing in for the simulator's dynamics.

State
-----
:class:`ModelState` carries only what changes within a decision — per-robot poses
and the last applied action (which feeds the GAT node features).  Goals, obstacle
geometry and all constants live on :class:`ForwardModel`, rebuilt once per real
decision (receding horizon).
"""

from __future__ import annotations

import hashlib
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


def _state_group_seed(poses: np.ndarray, group: int) -> int:
    """
    Stable 32-bit seed for the coarse column drop, keyed by (poses, group).

    Rounding to micrometres makes the seed insensitive to float noise while
    remaining state-dependent, so the same node always vets the same coarse plan.
    """
    key = np.ascontiguousarray(np.round(poses, 6)).tobytes()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ (group * 2654435761)) & 0xFFFFFFFF


@dataclass
class ModelState:
    """Dynamic planner state for one lookahead node."""

    poses: np.ndarray         # (N, 3) [x, y, theta]
    last_actions: np.ndarray  # (N, 2) last applied sim-input [lin, ang]


@dataclass
class CoarseMove:
    """A vetted coarse action at a node: its candidate + the state it leads to."""

    candidate: CoarseCandidate  # shield dataclass (group, frames, clearance, progress, safe)
    next_state: ModelState


class ForwardModel:
    """
    Deterministic analytic model over robot poses for the MPC lookahead.

    Args:
        backbone:           Frozen ``GATBackbone`` (precise actions + state prep).
        coarse:             ``CoarseSteering`` primitive (shared with the env).
        goals:              (N, 2) fixed goal positions for the episode.
        obstacle_states:    (M, 4) obstacle observations for the GAT forward
                            (frozen within a decision).
        geom:               ``ShieldGeometry`` (robot radius + obstacle geometry)
                            for the safety mask and collision prediction.
        step_time:          Simulator sub-step duration (s); the integrator's dt.
        selection_interval: Sub-steps each robot is driven per precise decision.
        lin_max:            Max linear velocity (m/s) — used by the cost-to-go α.
        d_safe:             Clearance margin (m) a coarse move must keep to be safe.
        goal_threshold:     Per-robot goal-arrival radius (m) for ``all_reached``.
        reward_fn:          ``PathCostReward`` supplying the per-decision motion cost
                            (defaults to the eval configuration).
        leaf_value:         Optional learned leaf evaluator ``(model, ms) -> float``
                            (e.g. :class:`LearnedCostToGo`).  When set,
                            :meth:`cost_to_go` dispatches to it instead of the
                            crude analytic heuristic.
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

        # Expansion counters for budget accounting across search algorithms.
        # A node expansion runs coarse_moves + precise_next exactly once each,
        # so ``n_precise_expansions`` equals the number of expanded nodes.
        self.n_precise_expansions = 0
        self.n_coarse_vets = 0

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    @staticmethod
    def state_from_robot_state(robot_state: np.ndarray) -> ModelState:
        """
        Recover a :class:`ModelState` from an 11-col ``robot_state`` row block.

        The GAT state is invertible: heading is ``atan2(sinθ, cosθ)`` and the last
        action is undone from the node-feature scalings used by
        ``TD3Obstacle.prepare_state`` (``lin_vel = act0·2``, ``ang_vel = (act1+1)/2``).
        This lets ``decide(robot_state)`` build the root node self-contained, exactly
        matching what the backbone saw.
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
        Build the (N, 11) GAT state for ``ms`` — mirrors ``sim.step`` geometry then
        ``backbone.prepare_state`` so the layout/scaling exactly match training.
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
        Geometric collision prediction for a node's state.

        Robot–robot: any centre distance < 2ρ.  Robot–obstacle: any clearance
        (centre distance − r_obs − ρ) < 0.  Same geometry the shield uses.
        """
        pos = ms.poses[:, :2]
        rho = self.geom.rho
        # robot–robot
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        if d.min() < 2.0 * rho:
            return True
        # robot–obstacle
        if self.geom.obstacle_xy.shape[0] > 0:
            od = np.linalg.norm(
                pos[:, None, :] - self.geom.obstacle_xy[None, :, :], axis=2
            )
            clr = od - self.geom.obstacle_r[None, :] - rho
            if clr.min() < 0.0:
                return True
        return False

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def coarse_moves(self, ms: ModelState) -> dict[int, CoarseMove]:
        """
        Vet and expand every selectable coarse group at ``ms``.

        Returns a mapping ``group -> CoarseMove`` holding the shield candidate
        (clearance/progress/safe + the exact seeded frames to execute) and the
        deterministic next state the move leads to.
        """
        self.n_coarse_vets += 1
        rs = self.robot_state(ms)
        moves: dict[int, CoarseMove] = {}
        for group in self.coarse.selectable_groups():
            seed = _state_group_seed(ms.poses, group)
            rot, trans = self.coarse.compute_actions(rs, group, seed=seed)
            samples = swept_positions(rs, rot, trans, self.step_time)  # (T+1, N, 2)
            members = self.coarse.members_of(group)
            clearance = min_member_clearance(samples, members, self.geom)
            progress = predicted_progress(rs, members, samples[-1])
            candidate = CoarseCandidate(
                group=group,
                frames=rot + trans,
                clearance=clearance,
                progress=progress,
                safe=clearance >= self.d_safe,
            )
            moves[group] = CoarseMove(
                candidate=candidate,
                next_state=self._coarse_next_state(ms, rot, trans, samples[-1]),
            )
        return moves

    def _coarse_next_state(
        self,
        ms: ModelState,
        rotation_frames: list,
        translation_frames: list,
        end_positions: np.ndarray,
    ) -> ModelState:
        """
        Next state after a coarse control: end positions from the swept path, and
        the post-rotation heading (rotation frames only change θ; translation frames
        keep it fixed).  ``last_actions`` = the final executed frame, as the sim
        would report it to the next decision.
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
        Next state after one precise-all decision, mirroring
        :meth:`SwitcherEnv._run_precise`: resolve robots one at a time, each driven
        by its frozen GAT action for ``selection_interval`` sub-steps while the
        others hold still (receive ``[0, 0]``), integrating unicycle dynamics.
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
        """
        One unicycle sub-step for all robots (forward Euler, pre-update heading).

        Robots with zero linear/angular command are unchanged. Matched against the
        real simulator by the model-fidelity check; switch to a semi-implicit update
        there if irsim integrates θ before translating.
        """
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
        Per-decision motion cost = ``−PathCostReward`` for a non-terminal decision,
        so the planner minimises exactly the eval objective.

        Coarse ⇒ ``n_members(group) · move_distance``; precise ⇒ flat ``precise_cost``.
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
        Leaf cost-to-go.

        With a configured ``leaf_value`` (learned per-robot precise cost-to-go,
        summed over unreached robots) that evaluator is used and ``alpha`` is
        ignored.  Otherwise the crude analytic heuristic — finish precisely, one
        robot at a time:

        ``H(x) = α · Σ_i ‖p_i − goal_i‖`` with default
        ``α = precise_cost / (lin_max · step_time)`` — ``precise_cost`` charged per
        sub-step, one sub-step advancing a robot ``lin_max · step_time`` metres.
        """
        if self.leaf_value is not None:
            return float(self.leaf_value(self, ms))
        if alpha is None:
            alpha = self.reward_fn.precise_cost / (self.lin_max * self.step_time)
        return float(alpha) * float(np.sum(self.goal_distances(ms)))


# robot_state goal columns (see CoarseSteering.compute_actions docstring).
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
) -> ForwardModel:
    """
    Rebuild the deterministic forward model from the live sim + root state.

    The one place a switcher turns (backbone, coarse, live sim, root
    robot_state) into a :class:`ForwardModel` — shared by ``MPCSwitcher`` and
    the tree-search switchers so they all plan over identical models.
    """
    s = np.asarray(robot_state, dtype=np.float64)
    goals = s[:, [_GX, _GY]]
    return ForwardModel(
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
