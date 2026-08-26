"""
Sim-free tests for the physics fix + per-group precise action space
(redesign §2–§3): coupled rotation in ``ForwardModel14`` transitions, the
``precise_group_next`` transition, config-aware ``expand_stubs``, and the
legacy-behaviour guarantees when nothing new is configured.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE
from robot_nav.models.MARL.capswitcher.rl.shield import ShieldGeometry
from robot_nav.models.MARL.capswitcher_14.configs import (
    N_ROBOTS,
    build_precise_groups,
    make_coarse_steering,
)
from robot_nav.models.MARL.capswitcher_14.policies.precise_coupling import (
    PreciseCoupling,
)
from robot_nav.models.MARL.capswitcher_14.configs import A_FULL
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    ForwardModel14,
    ModelState,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.common import expand_stubs

COST_YAML = "robot_nav/models/MARL/capswitcher_14/cost_14robots.yaml"


class FakeBackbone:
    """
    Minimal stand-in for the frozen GAT: ``prepare_state`` mirrors the real
    11-col layout/scaling and ``get_embedding_and_actions`` steers each robot
    straight toward its goal (raw actor space: lin=1 → sim 0.5 m/s,
    ang = clipped heading error).
    """

    def prepare_state(self, poses, distance, cos, sin, collision, action, goals):
        states = [
            [
                poses[i][0], poses[i][1],
                np.cos(poses[i][2]), np.sin(poses[i][2]),
                distance[i], cos[i], sin[i],
                action[i][0] * 2.0, (action[i][1] + 1.0) / 2.0,
                goals[i][0], goals[i][1],
            ]
            for i in range(len(poses))
        ]
        return states, None

    def get_embedding_and_actions(self, rs, obstacle_states):
        rs = np.asarray(rs, dtype=np.float64)
        theta = np.arctan2(rs[:, 3], rs[:, 2])
        bearing = np.arctan2(rs[:, 10] - rs[:, 1], rs[:, 9] - rs[:, 0])
        err = (bearing - theta + np.pi) % (2 * np.pi) - np.pi
        raw = np.stack([np.ones(rs.shape[0]), np.clip(err, -1.0, 1.0)], axis=1)
        return raw, None


def make_model(coupling=None, precise_groups=None, seed=0) -> tuple[ForwardModel14, ModelState]:
    rng = np.random.default_rng(seed)
    poses = np.column_stack([
        rng.uniform(2.0, 8.0, N_ROBOTS),
        rng.uniform(2.0, 8.0, N_ROBOTS),
        rng.uniform(-np.pi, np.pi, N_ROBOTS),
    ])
    goals = poses[:, :2] + rng.uniform(-3.0, 3.0, (N_ROBOTS, 2))
    cost = SwitcherCost.from_yaml(COST_YAML)
    model = ForwardModel14(
        backbone=FakeBackbone(),
        coarse=make_coarse_steering(move_distance=cost.move_distances),
        goals=goals,
        obstacle_states=np.empty((0, 4)),
        geom=ShieldGeometry(
            rho=0.05,  # tiny radius: keeps random states collision-free
            obstacle_xy=np.empty((0, 2)),
            obstacle_r=np.empty(0),
        ),
        step_time=0.3,
        cost=cost,
        coupling=coupling,
        precise_groups=precise_groups,
    )
    ms = ModelState(poses=poses, last_actions=np.zeros((N_ROBOTS, 2)))
    return model, ms


# ---------------------------------------------------------------------------
# Legacy behaviour guards (config A, no coupling)
# ---------------------------------------------------------------------------


def test_legacy_stubs_unchanged() -> None:
    model, ms = make_model()
    stubs = expand_stubs(model, ms)
    assert len(stubs) == 23
    assert [b.mode for b in stubs[:-1]] == [COARSE] * 22
    assert stubs[-1].mode == PRECISE and stubs[-1].pgroup is None


def test_legacy_precise_keeps_bystanders_still() -> None:
    model, ms = make_model(coupling=None)
    nxt = model.precise_next(ms)
    # Without coupling, driven robots move one at a time and nobody else's
    # heading changes *while others are driven* — but every unreached robot is
    # eventually driven, so compare against a single-robot check instead:
    # drive with all robots reached except robot 0.
    model2, ms2 = make_model(coupling=None)
    model2.goals = ms2.poses[:, :2].copy()          # everyone at goal...
    model2.goals[0] += np.array([3.0, 0.0])         # ...except robot 0
    nxt2 = model2.precise_next(ms2)
    moved = np.abs(nxt2.poses[:, 2] - ms2.poses[:, 2]) > 1e-12
    assert moved[0] and not moved[1:].any()


# ---------------------------------------------------------------------------
# Coupled physics (fix §2)
# ---------------------------------------------------------------------------


def test_coupled_precise_rotates_bystanders() -> None:
    coupling = PreciseCoupling(A_FULL, ang_max=1.0)
    model, ms = make_model(coupling=coupling)
    model.goals = ms.poses[:, :2].copy()
    model.goals[0] += np.array([-3.0, 2.0])         # only robot 0 unreached
    nxt = model.precise_next(ms)
    dtheta = np.abs(
        (nxt.poses[:, 2] - ms.poses[:, 2] + np.pi) % (2 * np.pi) - np.pi
    )
    assert dtheta[0] > 1e-6                          # driven robot turned
    assert (dtheta[1:] > 1e-6).sum() >= 10           # bystanders side-rotated
    # ...but bystanders never translate.
    np.testing.assert_allclose(
        nxt.poses[1:, :2], ms.poses[1:, :2], atol=1e-12
    )


def test_precise_group_pair_drives_both_simultaneously() -> None:
    coupling = PreciseCoupling(A_FULL, ang_max=1.0)
    pgs = build_precise_groups("pairs")
    model, ms = make_model(coupling=coupling, precise_groups=pgs)
    nxt = model.precise_group_next(ms, 0)            # drive robots {0, 1}
    moved = np.linalg.norm(nxt.poses[:, :2] - ms.poses[:, :2], axis=1) > 1e-9
    assert moved[0] and moved[1] and not moved[2:].any()
    assert model.n_precise_expansions == 1


def test_precise_group_skips_reached_members() -> None:
    coupling = PreciseCoupling(A_FULL, ang_max=1.0)
    pgs = build_precise_groups("pairs")
    model, ms = make_model(coupling=coupling, precise_groups=pgs)
    model.goals[1] = ms.poses[1, :2]                 # robot 1 already reached
    assert model.driven_members(ms, 0).tolist() == [0]
    nxt = model.precise_group_next(ms, 0)
    assert np.linalg.norm(nxt.poses[1, :2] - ms.poses[1, :2]) < 1e-12


# ---------------------------------------------------------------------------
# Config-aware stubs + costs (§3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config,n_precise", [("pairs", 7), ("singles", 14)])
def test_stub_counts_per_config(config: str, n_precise: int) -> None:
    model, ms = make_model(precise_groups=build_precise_groups(config))
    stubs = expand_stubs(model, ms)
    precise = [b for b in stubs if b.mode == PRECISE]
    assert len(stubs) == 22 + n_precise
    assert len(precise) == n_precise
    assert [b.pgroup for b in precise] == list(range(n_precise))


def test_fully_reached_group_has_no_edge() -> None:
    pgs = build_precise_groups("pairs")
    model, ms = make_model(precise_groups=pgs)
    model.goals[0] = ms.poses[0, :2]
    model.goals[1] = ms.poses[1, :2]                 # pair 0 fully reached
    stubs = expand_stubs(model, ms)
    assert [b.pgroup for b in stubs if b.mode == PRECISE] == list(range(1, 7))


def test_precise_group_step_cost_scales_with_driven() -> None:
    pgs = build_precise_groups("pairs")
    model, ms = make_model(precise_groups=pgs)
    unit, si = model.cost.precise_unit, model.selection_interval
    assert model.step_cost(PRECISE, ms, pgroup=0) == pytest.approx(2 * unit * si)
    model.goals[1] = ms.poses[1, :2]                 # one member reached
    assert model.step_cost(PRECISE, ms, pgroup=0) == pytest.approx(1 * unit * si)


def test_best_first_runs_with_precise_groups() -> None:
    """A* over config C materialises per-group precise edges end to end."""
    from robot_nav.models.MARL.capswitcher_14.rl.search.best_first import (
        BestFirstSearch14,
        EVALUATIONS,
    )

    from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
    from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
        CoarseSteering14,
    )

    coupling = PreciseCoupling(A_FULL, ang_max=1.0)
    pgs = build_precise_groups("singles")
    model, ms = make_model(coupling=coupling, precise_groups=pgs, seed=3)
    # Shrink the problem so the capped search finishes fast: only 3 coarse
    # groups, and everyone at goal except two robots whose goals lie straight
    # ahead of their current headings (the fake GT policy walks right in).
    model.coarse = CoarseSteering14(
        A_full=A_FULL, move_groups=MOVE_GROUPS[:3], move_distance=0.5
    )
    model.goals = ms.poses[:, :2].copy()
    for r in (0, 5):
        model.goals[r] = ms.poses[r, :2] + 0.8 * np.array(
            [np.cos(ms.poses[r, 2]), np.sin(ms.poses[r, 2])]
        )
    search = BestFirstSearch14(EVALUATIONS["astar"], max_transitions=400)
    res = search.run(model, ms)
    assert res.decisions, "search returned an empty plan"
    for d in res.decisions:
        assert set(d) >= {"mode", "group", "pgroup", "frames"}
        if d["mode"] == PRECISE:
            assert d["pgroup"] in range(len(pgs))
    if res.solved and not res.cap_hit:
        # Replaying the plan through the model must reach the goal state.
        cur = ms
        for d in res.decisions:
            if d["mode"] == PRECISE:
                cur = model.precise_group_next(cur, d["pgroup"])
            else:
                mv = model.coarse_move(cur, d["group"])
                assert mv.candidate.safe
                cur = mv.next_state
        assert model.all_reached(cur)


def test_group_rollout_advances_toward_goals() -> None:
    coupling = PreciseCoupling(A_FULL, ang_max=1.0)
    pgs = build_precise_groups("singles")
    model, ms = make_model(coupling=coupling, precise_groups=pgs)
    d0 = model.goal_distances(ms)
    # Pick a robot already facing its goal: the fake backbone turns at most
    # ang_max per sub-step while always driving forward, so a robot facing
    # away legitimately loses ground during its first decision.
    bearing = np.arctan2(
        model.goals[:, 1] - ms.poses[:, 1], model.goals[:, 0] - ms.poses[:, 0]
    )
    err = np.abs((bearing - ms.poses[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    r = int(np.argmin(err + np.where(d0 > 1.0, 0.0, 10.0)))  # aligned & unreached
    nxt = model.precise_group_next(ms, r)
    d1 = model.goal_distances(nxt)
    assert d1[r] < d0[r]                             # the GT proxy makes progress
