"""Tests for search-trace recording + offline label extraction (redesign §6)."""

from __future__ import annotations

import json

import numpy as np

from robot_nav.models.MARL.capswitcher_14.configs import (
    A_FULL,
    MOVE_GROUPS,
    build_precise_groups,
)
from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
    CoarseSteering14,
)
from robot_nav.models.MARL.capswitcher_14.policies.precise_coupling import (
    PreciseCoupling,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.best_first import (
    EVALUATIONS,
    BestFirstSearch14,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.trace import (
    TraceRecorder,
    load_plan,
    on_path_rows,
    value_labels,
)

from tests.test_precise_groups_14 import make_model


def _solved_traced_plan(tmp_path):
    """Run the small solvable search from the search test, recorded."""
    coupling = PreciseCoupling(A_FULL, ang_max=1.0)
    pgs = build_precise_groups("singles")
    model, ms = make_model(coupling=coupling, precise_groups=pgs, seed=3)
    model.coarse = CoarseSteering14(
        A_full=A_FULL, move_groups=MOVE_GROUPS[:3], move_distance=0.5
    )
    model.goals = ms.poses[:, :2].copy()
    for r in (0, 5):
        model.goals[r] = ms.poses[r, :2] + 0.8 * np.array(
            [np.cos(ms.poses[r, 2]), np.sin(ms.poses[r, 2])]
        )
    recorder = TraceRecorder(tmp_path / "shard", meta={"algo": "astar"})
    recorder.set_episode(episode=0, seed=123)
    search = BestFirstSearch14(
        EVALUATIONS["astar"], max_transitions=400, recorder=recorder
    )
    res = search.run(model, ms)
    return res, tmp_path / "shard"


def test_trace_shard_written_and_consistent(tmp_path) -> None:
    res, shard = _solved_traced_plan(tmp_path)
    assert json.loads((shard / "meta.json").read_text())["algo"] == "astar"
    files = sorted(shard.glob("plan_*.npz"))
    assert len(files) == 1
    plan = load_plan(files[0])

    m = plan["g"].shape[0]                       # expanded nodes
    assert plan["poses"].shape == (m, 14, 3)
    assert plan["last_actions"].shape == (m, 14, 2)
    assert int(plan["expansions"]) == m
    assert int(plan["episode"]) == 0 and int(plan["seed"]) == 123
    # root is row 0 with no parent
    assert plan["parent_row"][0] == -1 and plan["depth"][0] == 0
    # every branch row points at a valid node row
    assert plan["br_node"].min() >= 0 and plan["br_node"].max() < m
    # coarse rows carry clearance; precise rows carry NaN clearance
    coarse_rows = plan["br_mode"] == 0
    assert np.isfinite(plan["br_clearance"][coarse_rows]).all()
    assert np.isnan(plan["br_clearance"][~coarse_rows]).all()
    # step costs are positive and children's g = parent g + step cost
    gen = np.isfinite(plan["br_child_g"])
    np.testing.assert_allclose(
        plan["br_child_g"][gen],
        plan["g"][plan["br_node"][gen]] + plan["br_step_cost"][gen],
        rtol=1e-6,
    )


def test_value_and_prior_labels_from_solved_plan(tmp_path) -> None:
    res, shard = _solved_traced_plan(tmp_path)
    plan = load_plan(next(shard.glob("plan_*.npz")))
    assert bool(plan["solved"]) == res.solved
    if not res.solved:      # cap-hit fallback: no labels, and that's correct
        rows, y = value_labels(plan)
        assert rows.size == 0
        return
    np.testing.assert_allclose(float(plan["plan_cost"]), res.plan_cost, rtol=1e-6)
    rows, aidx = on_path_rows(plan)
    # Path length = number of plan decisions (goal state itself is unexpanded).
    assert rows.size == len(res.decisions)
    assert rows[0] == 0                          # starts at the root
    # Depths increase along the path; chosen branches were legal.
    assert (np.diff(plan["depth"][rows]) == 1).all()
    rows_v, y = value_labels(plan)
    np.testing.assert_array_equal(rows_v, rows)
    # Remaining cost decreases strictly along the path and starts at plan_cost.
    assert y[0] == float(plan["plan_cost"])
    assert (np.diff(y) < 0).all()
    # y(node) - y(child on path) equals the chosen branch's step cost.
    for i in range(rows.size - 1):
        mask = plan["br_node"] == rows[i]
        step = plan["br_step_cost"][mask][aidx[i]]
        np.testing.assert_allclose(y[i] - y[i + 1], step, rtol=1e-5)
