"""
Contract tests for :class:`SwitcherCost` and the shipped cost YAMLs.

The shipped costs are free constants the user edits, so these tests pin the
*mechanism* (pricing formulas, drift validation, schema) and check the two
YAMLs against the live group algebra — never the specific cost values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.configs import (
    MOVE_GROUPS,
    make_coarse_steering,
)

REPO = Path(__file__).resolve().parents[1]
COST_6 = REPO / "robot_nav/models/MARL/capswitcher/cost_6robots.yaml"
COST_14 = REPO / "robot_nav/models/MARL/capswitcher_14/cost_14robots.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cost.yaml"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# Pricing formulas
# ---------------------------------------------------------------------------


def test_precise_cost_is_unit_times_robots_times_substeps(tmp_path):
    cost = SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 2.5
coarse:
  - {id: 0, members: [0, 1], move_distance: 1.0, cost: 7.0}
"""))
    assert cost.precise_cost(6, 5) == pytest.approx(2.5 * 6 * 5)
    assert cost.precise_cost(14, 5) == pytest.approx(2.5 * 14 * 5)
    # Reached robots are simply not counted.
    assert cost.precise_cost(0, 5) == 0.0


def test_precise_substep_cost_matches_per_robot_accounting(tmp_path):
    """One robot moves per precise sub-step, so both routes must agree."""
    cost = SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 3.0
coarse:
  - {id: 0, members: [0], move_distance: 1.0, cost: 1.0}
"""))
    assert cost.precise_substep_cost(9 * 5) == pytest.approx(cost.precise_cost(9, 5))


def test_coarse_cost_and_move_distance_are_per_group(tmp_path):
    cost = SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 1.0
coarse:
  - {id: 0, members: [0, 1, 2], move_distance: 1.5, cost: 4.5}
  - {id: 1, members: [3, 4, 5, 6], move_distance: 0.8, cost: 99.0}
"""))
    assert cost.coarse_cost(0) == pytest.approx(4.5)
    assert cost.coarse_cost(1) == pytest.approx(99.0)
    assert cost.move_distance(0) == pytest.approx(1.5)
    assert cost.move_distance(1) == pytest.approx(0.8)
    assert cost.move_distances == {0: 1.5, 1: 0.8}
    assert cost.group_ids == [0, 1]


def test_cost_is_free_of_the_members_times_distance_formula(tmp_path):
    """A configured cost must survive verbatim, however unrelated to geometry."""
    cost = SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 1.0
coarse:
  - {id: 0, members: [0, 1, 2], move_distance: 1.5, cost: 0.25}
"""))
    assert cost.coarse_cost(0) == pytest.approx(0.25)   # not 3 * 1.5


def test_auto_cost_resolves_to_members_times_distance(tmp_path):
    cost = SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 1.0
coarse:
  - {id: 0, members: [0, 1, 2], move_distance: 1.5, cost: auto}
  - {id: 1, members: [3, 4, 5, 6, 7, 8, 9], move_distance: 1.0, cost: auto}
"""))
    assert cost.coarse_cost(0) == pytest.approx(3 * 1.5)
    assert cost.coarse_cost(1) == pytest.approx(7 * 1.0)


# ---------------------------------------------------------------------------
# Schema + drift validation
# ---------------------------------------------------------------------------


def test_missing_top_level_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="precise_unit"):
        SwitcherCost.from_yaml(_write(tmp_path, "coarse: []\n"))


def test_empty_group_list_raises(tmp_path):
    with pytest.raises(ValueError, match="at least one group"):
        SwitcherCost.from_yaml(_write(tmp_path, "precise_unit: 1.0\ncoarse: []\n"))


def test_duplicate_group_ids_raise(tmp_path):
    with pytest.raises(ValueError, match="duplicate group ids"):
        SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 1.0
coarse:
  - {id: 0, members: [0], move_distance: 1.0, cost: 1.0}
  - {id: 0, members: [1], move_distance: 1.0, cost: 1.0}
"""))


def test_validate_members_detects_reordering(tmp_path):
    cost = SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 1.0
coarse:
  - {id: 0, members: [0, 1, 2], move_distance: 1.0, cost: 1.0}
  - {id: 1, members: [3, 4, 5], move_distance: 1.0, cost: 1.0}
"""))
    cost.validate_members({0: [0, 1, 2], 1: [3, 4, 5]})          # matching
    with pytest.raises(ValueError, match="members"):
        cost.validate_members({0: [3, 4, 5], 1: [0, 1, 2]})      # swapped


def test_validate_members_detects_group_count_mismatch(tmp_path):
    cost = SwitcherCost.from_yaml(_write(tmp_path, """
precise_unit: 1.0
coarse:
  - {id: 0, members: [0], move_distance: 1.0, cost: 1.0}
"""))
    with pytest.raises(ValueError, match="group ids"):
        cost.validate_members({0: [0], 1: [1]})


# ---------------------------------------------------------------------------
# The shipped configs match the live group algebra
# ---------------------------------------------------------------------------


def test_shipped_14robot_config_matches_move_groups():
    cost = SwitcherCost.from_yaml(COST_14)
    assert len(cost.group_ids) == len(MOVE_GROUPS) == 22
    cost.validate_members({i: list(g) for i, g in enumerate(MOVE_GROUPS)})


def test_shipped_6robot_config_matches_coarse_steering():
    cost = SwitcherCost.from_yaml(COST_6)
    cs = CoarseSteering(num_robots=6, move_distance=cost.move_distances)
    assert sorted(cost.group_ids) == sorted(cs.selectable_groups())
    cost.validate_members(
        {g: [int(i) for i in cs.members_of(g)] for g in cs.selectable_groups()}
    )


def test_shipped_14robot_move_distances_drive_the_primitive():
    """The YAML's per-group distances are what the members actually travel."""
    cost = SwitcherCost.from_yaml(COST_14)
    cs = make_coarse_steering(move_distance=cost.move_distances, step_time=0.3)

    state = np.zeros((14, 11))
    state[:, 0] = np.arange(14) * 0.4
    state[:, 2] = 1.0            # facing +x
    state[:, 9] = 30.0           # goal far east so headings barely change
    state[:, 10] = 0.0

    for group in (0, len(MOVE_GROUPS) - 1):
        _, trans = cs.compute_actions(state, group)
        travelled = np.zeros(14)
        for frame in trans:
            travelled += np.array([pair[0] for pair in frame]) * cs.step_time
        members = cs.members_of(group)
        assert np.allclose(travelled[members], cost.move_distance(group))
        non_members = np.setdiff1d(np.arange(14), members)
        assert np.allclose(travelled[non_members], 0.0)
