"""
Sim-free tests for the leaf cost-to-go labelling pipeline (stage 1).

The label is a suffix sum with a censoring rule — cheap to write, easy to get
subtly wrong, and wrong in a way no downstream metric would flag: an off-by-one
in the alignment or a leaked truncated episode just makes the fitted correction
quietly optimistic.  These pin the alignment, the censoring and the shared
analytic heuristic.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.analyze_leaf_data import _explained, _log_spread, report
from robot_nav.collect_leaf_data import (
    backfill_returns,
    goal_distances_from_state,
)
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    analytic_alpha,
    analytic_cost_to_go,
)


def _episode(index, step_costs, reached, h=None, n_unreached=None):
    k = len(step_costs)
    return {
        "index": index,
        "step_costs": list(step_costs),
        "h": list(h if h is not None else [100.0] * k),
        "n_unreached": list(n_unreached if n_unreached is not None else [14] * k),
        "sum_dist": [10.0] * k,
        "mode": [0] * k,
        "group": [3] * k,
        "group_feats": [np.zeros((22, 10), dtype=np.float32)] * k,
        "global_feats": [np.zeros(3, dtype=np.float32)] * k,
        "reached": reached,
    }


# ---------------------------------------------------------------------------
# The analytic heuristic (shared with the search)
# ---------------------------------------------------------------------------


def test_analytic_alpha_matches_precise_pricing():
    # precise_unit charged per robot per sub-step; a sub-step covers
    # lin_max * step_time metres.
    assert analytic_alpha(11.57, 0.5, 0.3) == pytest.approx(11.57 / 0.15)


def test_analytic_cost_to_go_skips_reached_robots():
    dist = np.array([0.1, 0.2, 5.0, 3.0])
    # Only the two beyond the threshold are charged.
    assert analytic_cost_to_go(dist, 0.3, 2.0) == pytest.approx(2.0 * 8.0)
    assert analytic_cost_to_go(np.array([0.1, 0.2]), 0.3, 2.0) == 0.0


def test_analytic_cost_to_go_matches_forward_model_formula():
    """The collector must score the same function the search plans with."""
    rng = np.random.default_rng(0)
    dist = rng.uniform(0.0, 8.0, size=14)
    thr, alpha = 0.3, 77.1
    expected = alpha * dist[dist > thr].sum()
    assert analytic_cost_to_go(dist, thr, alpha) == pytest.approx(expected)


def test_goal_distances_from_state_reads_the_right_columns():
    s = np.zeros((3, 11))
    s[:, 0], s[:, 1] = [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]     # px, py
    s[:, 9], s[:, 10] = [3.0, 1.0, 2.0], [4.0, 0.0, 0.0]    # gx, gy
    assert np.allclose(goal_distances_from_state(s), [5.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Return backfill
# ---------------------------------------------------------------------------


def test_return_is_suffix_sum_including_own_step():
    """G[t] must be the cost from state t onward — its own decision included."""
    data = backfill_returns([_episode(0, [1.0, 2.0, 4.0], reached=True)])
    assert np.allclose(data["G"], [7.0, 6.0, 4.0])
    # The last decision's return is exactly its own cost, not zero: the
    # heuristic at that state still has one decision left to pay for.
    assert data["G"][-1] == pytest.approx(4.0)


def test_return_alignment_survives_multiple_episodes():
    data = backfill_returns([
        _episode(0, [1.0, 2.0], reached=True),
        _episode(1, [5.0], reached=True),
        _episode(2, [1.0, 1.0, 1.0], reached=True),
    ])
    #                     ep0        ep1    ep2
    assert np.allclose(data["G"], [3.0, 2.0, 5.0, 3.0, 2.0, 1.0])
    assert np.array_equal(data["episode"], [0, 0, 1, 2, 2, 2])
    assert np.array_equal(data["t"], [0, 1, 0, 0, 1, 2])


def test_ratio_uses_the_logged_heuristic():
    data = backfill_returns(
        [_episode(0, [10.0, 30.0], reached=True, h=[400.0, 300.0])]
    )
    assert np.allclose(data["ratio"], [40.0 / 400.0, 30.0 / 300.0])


# ---------------------------------------------------------------------------
# Censoring — the rule that keeps the fit honest
# ---------------------------------------------------------------------------


def test_unsolved_episodes_are_dropped_by_default():
    data = backfill_returns([
        _episode(0, [1.0, 2.0], reached=True),
        _episode(1, [9.0, 9.0, 9.0], reached=False),   # timeout / collision
    ])
    assert data["G"].size == 2
    assert np.all(data["valid"])
    assert set(np.unique(data["episode"])) == {0}


def test_kept_censored_episodes_never_produce_a_ratio():
    """
    A truncated sum understates the true remaining cost.  Letting it into the
    fit would bias the correction toward optimism — the direction that makes
    the search waste budget — so `ratio` must stay NaN even when the rows are
    retained for inspection.
    """
    data = backfill_returns(
        [_episode(0, [9.0, 9.0], reached=False)], keep_censored=True
    )
    assert data["G"].size == 2
    assert not np.any(data["valid"])
    assert np.all(np.isnan(data["ratio"]))


def test_empty_and_all_censored_inputs_are_safe():
    assert backfill_returns([])["G"].size == 0
    assert backfill_returns([_episode(0, [], reached=True)])["G"].size == 0
    assert backfill_returns([_episode(0, [1.0], reached=False)])["G"].size == 0


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def test_log_spread_is_robust_to_outliers():
    x = np.concatenate([np.random.default_rng(0).normal(0.0, 1.0, 2000),
                        np.full(20, 500.0)])
    med, sigma = _log_spread(x)
    assert abs(med) < 0.15
    assert 0.8 < sigma < 1.2        # IQR-based: the outliers do not inflate it


def test_explained_variance_detects_and_ignores_structure():
    groups = np.repeat([0, 1], 100)
    # Fully explained by the grouping.
    assert _explained(np.repeat([1.0, 5.0], 100), groups) == pytest.approx(1.0)
    # Unrelated grouping explains ~nothing.
    x = np.random.default_rng(1).normal(size=200)
    assert _explained(x, groups) < 0.1


def test_report_runs_end_to_end_and_verdicts(capsys):
    """A tight ratio must verdict 'constant is enough'; a spread one must not."""
    rng = np.random.default_rng(0)
    n = 500

    def _shards(sigma):
        h = rng.uniform(1000.0, 5000.0, n)
        ratio = 0.1 * np.exp(rng.normal(0.0, sigma, n))
        return {
            "h": h, "G": h * ratio, "ratio": ratio,
            "n_unreached": rng.integers(1, 15, n),
            "sum_dist": h / 77.1, "mode": rng.integers(0, 2, n),
            "group": np.zeros(n, dtype=int), "valid": np.ones(n, dtype=bool),
            "episode": np.zeros(n, dtype=int), "t": np.arange(n),
            "alpha": np.float64(77.1), "n_shards": 1,
        }

    report(_shards(0.05))
    assert "VERDICT: a constant is enough" in capsys.readouterr().out

    report(_shards(0.9))
    assert "VERDICT: build the value head" in capsys.readouterr().out
