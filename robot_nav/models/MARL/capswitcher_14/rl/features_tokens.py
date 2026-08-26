"""
Token feature builder for the redesigned value/prior networks
(``docs/value_prior_redesign.md`` §4) — N-generic, configuration-generic,
numpy-only, sim-free.

One state produces four token groups:

* **robot tokens** (N, ``ROBOT_DIM``) — geometry, coupled-controllability
  (``P = A·A⁺`` features), obstacle clearances, robot/obstacle blocking along
  the goal ray, crowding, dynamics state;
* **edge features** (N, N, ``EDGE_DIM``) — relative geometry ⊕ coupling
  ``P_ij`` ⊕ on-ray blocking, the relational encoder's attention bias;
* **action tokens** (K, ``ACTION_DIM``) + type/membership — one per candidate
  edge (coarse move-groups + precise groups), carrying composition, the
  **rotation preview**, and cost features;
* **global token** (``GLOBAL_DIM``,).

Rotation preview tiers (validated by ``robot_nav/check_coupling_features.py``,
which falsified the pure-linear preview for the deployed nonlinear primitive):

* coarse groups — the **exact rotation solve** of the configured
  ``CoarseSteering14`` (its frames, no sweep/vet; ~0.4 ms/group);
* precise groups — the exact linear solve ``wrap(A @ pinv(A_S) @ dθ_S)``
  (that *is* the coupled-precise primitive's rotation).

Everything is normalized by explicit scales (``dist_scale``, ``clearance_cap``,
π) and free of absolute coordinates and fixed sizes — no feature depends on
N, K, or the group algebra beyond what ``A`` itself encodes.

Offline use: :meth:`TokenFeatureBuilder.from_shard_meta` rebuilds a builder
from a trace shard's ``meta.json`` (see ``rl/search/trace.py``); features are
then built from each stored node's poses/last_actions + the plan's goals and
obstacle geometry — no sim, no search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROBOT_DIM = 16
EDGE_DIM = 6
ACTION_DIM = 13
GLOBAL_DIM = 8

_AHEAD_HALF_ANGLE = np.pi / 3.0     # ±60° cone toward the goal bearing
_K_NEAREST = 3

TYPE_COARSE = 0
TYPE_PRECISE = 1


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class TokenFeatureBuilder:
    """
    Args:
        A_full:         (N, K_act) actuation matrix.
        coarse:         ``CoarseSteering14`` — supplies the move-groups and the
                        exact rotation solve for the coarse preview.
        precise_groups: list of member-index lists, or ``None`` (config "all":
                        a single precise token containing every robot).
        coarse_costs:   {group_id: step cost} (``SwitcherCost.coarse_cost``).
        precise_unit:   cost of one robot for one precise sub-step.
        selection_interval: sub-steps per precise decision.
        lin_max / step_time: bound the per-decision precise travel used by the
                        precise progress preview.
        goal_threshold: arrival radius (m).
        dist_scale:     distance normalizer (m), ~arena diameter.
        clearance_cap:  clearance saturation (m).
    """

    def __init__(
        self,
        A_full: np.ndarray,
        coarse,
        precise_groups: list | None,
        coarse_costs: dict,
        precise_unit: float,
        selection_interval: int = 5,
        lin_max: float = 0.5,
        step_time: float = 0.3,
        goal_threshold: float = 0.3,
        dist_scale: float = 10.0,
        clearance_cap: float = 2.0,
    ) -> None:
        self.A = np.asarray(A_full, dtype=np.float64)
        self.N = self.A.shape[0]
        self.P = self.A @ np.linalg.pinv(self.A)
        self.rankA = int(np.linalg.matrix_rank(self.A))
        self.coarse = coarse
        self.move_groups = [
            np.asarray(coarse.members_of(g), dtype=int)
            for g in coarse.selectable_groups()
        ]
        if precise_groups is None:
            self.precise_groups = [np.arange(self.N)]
        else:
            self.precise_groups = [np.asarray(g, dtype=int) for g in precise_groups]
        # Cached pinv per precise group for the linear preview.
        self._pinv_S = [np.linalg.pinv(self.A[g, :]) for g in self.precise_groups]
        self.coarse_costs = {int(k): float(v) for k, v in coarse_costs.items()}
        self.precise_unit = float(precise_unit)
        self.selection_interval = int(selection_interval)
        self.lin_max = float(lin_max)
        self.step_time = float(step_time)
        self.goal_threshold = float(goal_threshold)
        self.dist_scale = float(dist_scale)
        self.clearance_cap = float(clearance_cap)
        # One precise decision moves a driven robot at most this far.
        self._precise_travel = self.selection_interval * self.lin_max * self.step_time
        self._decision_price = self.precise_unit * self.selection_interval

    # ------------------------------------------------------------------

    @classmethod
    def from_shard_meta(cls, meta_path: str | Path, cost) -> "TokenFeatureBuilder":
        """
        Rebuild the builder from a trace shard's ``meta.json`` + the cost
        table (coarse step costs are in ``cost_14robots.yaml``, not the meta).
        """
        from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
            CoarseSteering14,
        )

        meta = json.loads(Path(meta_path).read_text())
        A = np.asarray(meta["A_full"], dtype=np.float64)
        coarse = CoarseSteering14(
            A_full=A,
            move_groups=meta["move_groups"],
            move_distance={int(k): v for k, v in meta["move_distances"].items()},
            method="nonlinear",
        )
        return cls(
            A_full=A,
            coarse=coarse,
            precise_groups=meta["precise_groups"],
            coarse_costs={g: cost.coarse_cost(g)
                          for g in coarse.selectable_groups()},
            precise_unit=float(meta["precise_unit"]),
            selection_interval=int(meta["selection_interval"]),
            goal_threshold=float(meta["goal_threshold"]),
        )

    # ------------------------------------------------------------------
    # Per-state geometry helpers
    # ------------------------------------------------------------------

    def _corridor_clearance(
        self, pos: np.ndarray, goals: np.ndarray, centers: np.ndarray,
        radii: np.ndarray, self_mask: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For each robot i: clearance of every ``center`` to the segment
        ``pos_i → goals_i`` (perpendicular distance − radius, only where the
        center projects onto the segment; cap elsewhere).

        Returns ``(clear, on_seg)`` of shape (N, M): clearance per (robot,
        center) and whether the center lies over the segment.
        """
        n, m = pos.shape[0], centers.shape[0]
        seg = goals - pos                                    # (N, 2)
        seg_len = np.linalg.norm(seg, axis=1)                # (N,)
        safe_len = np.where(seg_len > 1e-9, seg_len, 1e-9)
        u = seg / safe_len[:, None]                          # (N, 2)
        rel = centers[None, :, :] - pos[:, None, :]          # (N, M, 2)
        t = np.einsum("nmd,nd->nm", rel, u)                  # projection length
        on_seg = (t > 0.0) & (t < seg_len[:, None])
        perp = np.abs(rel[..., 0] * u[:, None, 1] - rel[..., 1] * u[:, None, 0])
        clear = perp - radii[None, :]
        clear = np.where(on_seg, clear, self.clearance_cap)
        if self_mask and m == n:
            idx = np.arange(n)
            clear[idx, idx] = self.clearance_cap
            on_seg[idx, idx] = False
        return np.minimum(clear, self.clearance_cap), on_seg

    # ------------------------------------------------------------------

    def __call__(
        self,
        poses: np.ndarray,
        last_actions: np.ndarray,
        goals: np.ndarray,
        obstacle_xy: np.ndarray,
        obstacle_r: np.ndarray,
        rho: float,
    ) -> dict:
        """
        Build all token features for one state.

        Returns dict with ``robot`` (N, ROBOT_DIM), ``edge`` (N, N, EDGE_DIM),
        ``action`` (K, ACTION_DIM), ``action_type`` (K,), ``action_members``
        (K, N) bool, ``glob`` (GLOBAL_DIM,) — all float32 (types int8/bool).
        """
        poses = np.asarray(poses, dtype=np.float64)
        last = np.asarray(last_actions, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        obstacle_xy = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1, 2)
        obstacle_r = np.asarray(obstacle_r, dtype=np.float64).reshape(-1)
        n = poses.shape[0]
        pos, theta = poses[:, :2], poses[:, 2]

        gvec = goals - pos
        dist = np.linalg.norm(gvec, axis=1)
        bearing = np.arctan2(gvec[:, 1], gvec[:, 0])
        err = _wrap(bearing - theta)                          # desired turn dθ
        arrived = dist <= self.goal_threshold

        # -- coupling features ------------------------------------------
        achieved = self.P @ err                               # (P dθ)_i
        residual = _wrap(err - achieved)                      # (I−P) dθ, wrapped

        # -- clearances -------------------------------------------------
        dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        static_rr = dmat.min(axis=1) - 2.0 * rho
        if obstacle_xy.shape[0] > 0:
            od = np.linalg.norm(pos[:, None, :] - obstacle_xy[None, :, :], axis=2)
            static_ro = (od - obstacle_r[None, :] - rho).min(axis=1)
            obearing = np.arctan2(
                obstacle_xy[None, :, 1] - pos[:, None, 1],
                obstacle_xy[None, :, 0] - pos[:, None, 0],
            )
            in_cone = np.abs(_wrap(obearing - bearing[:, None])) <= _AHEAD_HALF_ANGLE
            oc = od - obstacle_r[None, :] - rho
            ahead = np.where(in_cone, oc, np.inf).min(axis=1)
            ahead = np.minimum(ahead, self.clearance_cap)
        else:
            static_ro = np.full(n, self.clearance_cap)
            ahead = np.full(n, self.clearance_cap)
        static = np.minimum(np.minimum(static_rr, static_ro), self.clearance_cap)

        # -- blocking along the goal ray --------------------------------
        rr_clear, rr_onseg = self._corridor_clearance(
            pos, goals, pos, np.full(n, 2.0 * rho), self_mask=True
        )
        blocker = rr_clear.argmin(axis=1)
        rr_min = rr_clear.min(axis=1)
        blocker_arrived = arrived[blocker] & (rr_min < self.clearance_cap)
        n_corridor = (rr_clear < 0.0).sum(axis=1)
        if obstacle_xy.shape[0] > 0:
            ro_clear, _ = self._corridor_clearance(
                pos, goals, obstacle_xy, obstacle_r + rho, self_mask=False
            )
            ro_min = ro_clear.min(axis=1)
        else:
            ro_min = np.full(n, self.clearance_cap)

        # -- crowding ---------------------------------------------------
        k = min(_K_NEAREST, n - 1)
        crowd = np.sort(dmat, axis=1)[:, :k].mean(axis=1) if k > 0 else np.full(n, np.inf)

        cap = self.clearance_cap
        robot = np.stack([
            dist / self.dist_scale,
            np.cos(err), np.sin(err),
            arrived.astype(np.float64),
            np.diag(self.P),
            achieved / np.pi,
            np.abs(residual) / np.pi,
            static / cap,
            ahead / cap,
            rr_min / cap,
            blocker_arrived.astype(np.float64),
            n_corridor / n,
            ro_min / cap,
            np.minimum(crowd, self.dist_scale) / self.dist_scale,
            last[:, 0] / max(self.lin_max, 1e-9),
            last[:, 1],
        ], axis=1).astype(np.float32)

        # -- edges ------------------------------------------------------
        rel = pos[None, :, :] - pos[:, None, :]               # j from i
        edist = np.linalg.norm(rel, axis=2)
        ebear = np.arctan2(rel[..., 1], rel[..., 0]) - bearing[:, None]
        edge = np.stack([
            np.minimum(edist, self.dist_scale) / self.dist_scale,
            np.cos(ebear), np.sin(ebear),
            np.broadcast_to(arrived[None, :], (n, n)).astype(np.float64),
            rr_clear / cap,
            self.P,
        ], axis=2).astype(np.float32)

        # -- action tokens ----------------------------------------------
        rs11 = np.zeros((n, 11))
        rs11[:, 0], rs11[:, 1] = pos[:, 0], pos[:, 1]
        rs11[:, 2], rs11[:, 3] = np.cos(theta), np.sin(theta)
        rs11[:, 9], rs11[:, 10] = goals[:, 0], goals[:, 1]

        tokens, types, members_mask = [], [], []
        for g, members in enumerate(self.move_groups):
            rot, _trans = self.coarse.compute_actions(rs11, g)    # exact solve
            dtheta = np.zeros(n)
            for frame in rot:
                dtheta += np.asarray(frame)[:, 1] * self.coarse.step_time
            dtheta = _wrap(dtheta)
            tokens.append(self._action_row(
                members, members, dtheta, self.coarse.move_distances[g],
                self.coarse_costs[g], pos, goals, dist, err, arrived,
                static, ahead,
            ))
            types.append(TYPE_COARSE)
            members_mask.append(np.isin(np.arange(n), members))
        for pg, members in enumerate(self.precise_groups):
            driven = members[~arrived[members]]
            dtheta = np.zeros(n)
            if driven.size > 0:
                if driven.size == members.size:
                    t = self._pinv_S[pg] @ err[members]
                else:
                    # pinv of a submatrix ≠ submatrix of pinv: solve for the
                    # actually-driven rows (cheap; K_act × |driven|).
                    t = np.linalg.pinv(self.A[driven, :]) @ err[driven]
                dtheta = _wrap(self.A @ t)
            step_cost = self._decision_price * driven.size
            tokens.append(self._action_row(
                members, driven, dtheta, self._precise_travel,
                max(step_cost, 1e-9), pos, goals, dist, err, arrived,
                static, ahead,
            ))
            types.append(TYPE_PRECISE)
            members_mask.append(np.isin(np.arange(n), members))

        glob = np.array([
            float((~arrived).mean()),
            dist.mean() / self.dist_scale,
            dist.max() / self.dist_scale,
            static.min() / cap,
            np.log(n) / np.log(64.0),
            self.rankA / n,
            float((arrived & (rr_clear < 0.0).any(axis=0)).mean()),
            float(np.abs(residual).mean() / np.pi),
        ], dtype=np.float32)

        return {
            "robot": robot,
            "edge": edge,
            "action": np.stack(tokens).astype(np.float32),
            "action_type": np.asarray(types, dtype=np.int8),
            "action_members": np.stack(members_mask),
            "glob": glob,
        }

    # ------------------------------------------------------------------

    def _action_row(
        self, members, driven, dtheta, travel, step_cost, pos, goals, dist,
        err, arrived, static, ahead,
    ) -> np.ndarray:
        """
        One action token's features from its previewed rotation ``dtheta``.

        ``members`` = the group (composition features); ``driven`` = who
        actually translates (coarse: all members; precise: unreached members).
        """
        n = pos.shape[0]
        members = np.asarray(members, dtype=int)
        driven = np.asarray(driven, dtype=int)
        bystanders = np.setdiff1d(np.arange(n), driven)
        frac_arr = float(arrived[members].mean()) if members.size else 1.0
        if driven.size > 0:
            m_err = np.abs(_wrap(err[driven] - dtheta[driven]))
            theta_new = np.arctan2(goals[driven, 1] - pos[driven, 1],
                                   goals[driven, 0] - pos[driven, 0]) - \
                _wrap(err[driven] - dtheta[driven])
            move = np.minimum(travel, dist[driven])   # can't overshoot usefully
            newpos = pos[driven] + move[:, None] * np.stack(
                [np.cos(theta_new), np.sin(theta_new)], axis=1
            )
            progress = float(np.sum(
                dist[driven] - np.linalg.norm(goals[driven] - newpos, axis=1)
            ))
            m_static = static[driven].min()
            m_ahead = ahead[driven].min()
        else:
            m_err = np.array([0.0])
            progress, m_static, m_ahead = 0.0, self.clearance_cap, self.clearance_cap
        by_abs = np.abs(dtheta[bystanders]) if bystanders.size else np.array([0.0])
        arr_all = np.flatnonzero(arrived)
        arr_churn = np.abs(dtheta[arr_all]).max() if arr_all.size else 0.0
        cap = self.clearance_cap
        return np.array([
            driven.size / n,
            np.log(max(driven.size, 1)) / np.log(max(n, 2)),
            frac_arr,
            m_err.mean() / np.pi,
            m_err.max() / np.pi,
            by_abs.mean() / np.pi,
            by_abs.max() / np.pi,
            arr_churn / np.pi,
            progress / max(step_cost, 1e-9) * self._decision_price,  # progress per precise-decision-equivalent
            m_static / cap,
            m_ahead / cap,
            step_cost / self._decision_price,
            travel / max(self.dist_scale, 1e-9),
        ])
