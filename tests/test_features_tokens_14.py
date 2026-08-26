"""Tests for the token feature builder (redesign §4)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.configs import (
    A_FULL,
    N_ROBOTS,
    build_precise_groups,
    make_coarse_steering,
)
from robot_nav.models.MARL.capswitcher_14.rl.features_tokens import (
    ACTION_DIM,
    EDGE_DIM,
    GLOBAL_DIM,
    ROBOT_DIM,
    TYPE_COARSE,
    TYPE_PRECISE,
    TokenFeatureBuilder,
)

COST_YAML = "robot_nav/models/MARL/capswitcher_14/cost_14robots.yaml"


def make_builder(config: str = "singles", method: str = "nonlinear") -> TokenFeatureBuilder:
    cost = SwitcherCost.from_yaml(COST_YAML)
    coarse = make_coarse_steering(move_distance=cost.move_distances, method=method)
    return TokenFeatureBuilder(
        A_full=A_FULL,
        coarse=coarse,
        precise_groups=build_precise_groups(config),
        coarse_costs={g: cost.coarse_cost(g) for g in coarse.selectable_groups()},
        precise_unit=cost.precise_unit,
    )


def random_state(seed: int = 0):
    rng = np.random.default_rng(seed)
    poses = np.column_stack([
        rng.uniform(2, 8, N_ROBOTS), rng.uniform(2, 8, N_ROBOTS),
        rng.uniform(-np.pi, np.pi, N_ROBOTS),
    ])
    goals = poses[:, :2] + rng.uniform(-3, 3, (N_ROBOTS, 2))
    last = np.zeros((N_ROBOTS, 2))
    obstacle_xy = rng.uniform(2, 8, (5, 2))
    obstacle_r = np.full(5, 0.5)
    return poses, last, goals, obstacle_xy, obstacle_r


def test_shapes_and_finiteness_per_config() -> None:
    poses, last, goals, oxy, orr = random_state()
    for config, n_precise in (("all", 1), ("pairs", 7), ("singles", 14)):
        fb = make_builder(config)
        out = fb(poses, last, goals, oxy, orr, rho=0.2)
        k = 22 + n_precise
        assert out["robot"].shape == (N_ROBOTS, ROBOT_DIM)
        assert out["edge"].shape == (N_ROBOTS, N_ROBOTS, EDGE_DIM)
        assert out["action"].shape == (k, ACTION_DIM)
        assert out["action_type"].shape == (k,)
        assert out["action_members"].shape == (k, N_ROBOTS)
        assert out["glob"].shape == (GLOBAL_DIM,)
        for key in ("robot", "edge", "action", "glob"):
            assert np.isfinite(out[key]).all(), f"non-finite in {key} ({config})"
        assert (out["action_type"][:22] == TYPE_COARSE).all()
        assert (out["action_type"][22:] == TYPE_PRECISE).all()


def test_coupling_features_match_projection() -> None:
    poses, last, goals, oxy, orr = random_state(1)
    fb = make_builder()
    out = fb(poses, last, goals, oxy, orr, rho=0.2)
    P = A_FULL @ np.linalg.pinv(A_FULL)
    np.testing.assert_allclose(out["robot"][:, 4], np.diag(P), atol=1e-6)
    # residual + achieved must reconstruct the desired turn (mod wrap)
    err = np.arctan2(goals[:, 1] - poses[:, 1], goals[:, 0] - poses[:, 0]) - poses[:, 2]
    err = (err + np.pi) % (2 * np.pi) - np.pi
    ach = out["robot"][:, 5] * np.pi
    np.testing.assert_allclose(ach, P @ err, atol=1e-5)
    # edge coupling channel is exactly P
    np.testing.assert_allclose(out["edge"][:, :, 5], P, atol=1e-6)


def test_coarse_preview_exact_under_least_squares() -> None:
    """With the LS primitive, the preview's member error must equal wrap((I−P)dθ)."""
    poses, last, goals, oxy, orr = random_state(2)
    fb = make_builder(method="least_squares")
    out = fb(poses, last, goals, oxy, orr, rho=0.2)
    P = A_FULL @ np.linalg.pinv(A_FULL)
    err = np.arctan2(goals[:, 1] - poses[:, 1], goals[:, 0] - poses[:, 0]) - poses[:, 2]
    err = (err + np.pi) % (2 * np.pi) - np.pi
    resid = np.abs((err - P @ err + np.pi) % (2 * np.pi) - np.pi)
    for g in range(22):
        members = np.flatnonzero(out["action_members"][g])
        expected_max = resid[members].max() / np.pi
        assert out["action"][g, 4] == pytest.approx(expected_max, abs=1e-5)


def test_precise_singles_preview_zero_member_error() -> None:
    """1-robot precise groups steer their target exactly: member error 0."""
    poses, last, goals, oxy, orr = random_state(3)
    fb = make_builder("singles")
    out = fb(poses, last, goals, oxy, orr, rho=0.2)
    precise = out["action"][22:]
    assert np.allclose(precise[:, 3], 0.0, atol=1e-6)   # mean member err
    assert np.allclose(precise[:, 4], 0.0, atol=1e-6)   # max member err
    # ...but bystander churn is nonzero (the coupling is visible)
    assert (precise[:, 6] > 0.01).any()


def test_blocking_feature_sees_parked_robot() -> None:
    """A robot parked on another's goal ray must show up in features."""
    n = N_ROBOTS
    poses = np.zeros((n, 3))
    poses[:, 0] = np.arange(n) * 3.0        # spread out on a line, heading +x
    poses[:, 1] = np.arange(n) * 3.0
    goals = poses[:, :2] + np.array([2.0, 0.0])
    # park robot 1 exactly between robot 0 and its goal
    poses[1, :2] = poses[0, :2] + np.array([1.0, 0.0])
    goals[1] = poses[1, :2]                 # robot 1 arrived (parked)
    fb = make_builder()
    out = fb(poses, np.zeros((n, 2)), goals, np.zeros((0, 2)), np.zeros(0), rho=0.2)
    r0 = out["robot"][0]
    assert r0[9] < 0.0 / fb.clearance_cap + 1e-6   # corridor clearance negative
    assert r0[10] == pytest.approx(1.0)            # nearest blocker is arrived
    # an unblocked robot far from everyone shows capped corridor clearance
    assert out["robot"][7 % n][9] == pytest.approx(1.0)