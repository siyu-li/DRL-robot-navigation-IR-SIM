"""
Geometry contract of the banded ("corridor") episode layout.

Sim-free: ``corridor_layout`` is pure numpy, so these run locally even though
anything that steps irsim does not.
"""

import random

import numpy as np
import pytest

from robot_nav.SIM_ENV.corridor_layout import (
    LEFT_TO_RIGHT,
    RIGHT_TO_LEFT,
    CorridorLayout,
    sample_obstacles,
    sample_starts,
)

X_RANGE = (0.0, 15.0)
Y_RANGE = (0.0, 15.0)
RADII = np.full(7, 0.7)          # the 14-robot worlds' obstacle set
N_ROBOTS = 14


def _obstacles(layout, seed):
    np.random.seed(seed)
    return sample_obstacles(RADII, X_RANGE, Y_RANGE, layout)


def _min_surface_gap(obs):
    d = np.linalg.norm(obs[:, None, :2] - obs[None, :, :2], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d - RADII[:, None] - RADII[None, :]))


def test_bands_are_disjoint_and_ordered():
    layout = CorridorLayout()
    layout.validate()
    sx0, sx1 = layout.start_band(X_RANGE, LEFT_TO_RIGHT)
    bx0, bx1 = layout.obstacle_band(X_RANGE)
    gx0, gx1 = layout.goal_band(X_RANGE, LEFT_TO_RIGHT)
    assert sx0 < sx1 < bx0 < bx1 < gx0 < gx1


def test_validate_rejects_overlapping_bands():
    with pytest.raises(ValueError, match="start band"):
        CorridorLayout(start_width=0.5).validate()
    with pytest.raises(ValueError, match="goal band"):
        CorridorLayout(goal_width=0.5).validate()
    with pytest.raises(ValueError, match="band must satisfy"):
        CorridorLayout(band=(0.6, 0.4)).validate()


@pytest.mark.parametrize("seed", range(25))
def test_obstacles_land_in_the_band_and_leave_a_gate(seed):
    """
    Every obstacle inside the middle band, and no two closer than ``min_gap``.

    The gap is what keeps the band a gate rather than a wall: at the default
    1.0 m a shielded coarse move (rho=0.2, d_safe=0.3) can thread any pair.
    """
    layout = CorridorLayout()
    obs = _obstacles(layout, seed)
    bx0, bx1 = layout.obstacle_band(X_RANGE)
    by0, by1 = layout.y_band(Y_RANGE)

    assert np.all((obs[:, 0] >= bx0) & (obs[:, 0] <= bx1))
    assert np.all((obs[:, 1] >= by0) & (obs[:, 1] <= by1))
    assert _min_surface_gap(obs) >= layout.min_gap


@pytest.mark.parametrize("seed", range(25))
def test_starts_are_in_the_start_band_and_mutually_clear(seed):
    layout = CorridorLayout()
    obs = _obstacles(layout, seed)
    random.seed(seed)
    starts = sample_starts(N_ROBOTS, X_RANGE, Y_RANGE, layout, obs[:, :2], RADII)

    sx0, sx1 = layout.start_band(X_RANGE, LEFT_TO_RIGHT)
    assert np.all((starts[:, 0] >= sx0) & (starts[:, 0] <= sx1))

    d = np.linalg.norm(starts[:, None, :2] - starts[None, :, :2], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 0.6


def test_starts_and_goals_sit_on_opposite_sides_of_the_gate():
    """The traverse must cross the obstacle band — that is the whole point."""
    layout = CorridorLayout()
    obs = _obstacles(layout, 0)
    random.seed(0)
    starts = sample_starts(N_ROBOTS, X_RANGE, Y_RANGE, layout, obs[:, :2], RADII)
    bx0, bx1 = layout.obstacle_band(X_RANGE)
    lo, hi = layout.goal_range_limits(X_RANGE, Y_RANGE, LEFT_TO_RIGHT)

    assert starts[:, 0].max() < bx0
    assert lo[0] > bx1


def test_bidirectional_splits_the_swarm_and_mirrors_its_goals():
    layout = CorridorLayout(bidirectional=True)
    directions = layout.directions(N_ROBOTS)
    assert (directions == LEFT_TO_RIGHT).sum() == N_ROBOTS // 2
    assert (directions == RIGHT_TO_LEFT).sum() == N_ROBOTS - N_ROBOTS // 2

    obs = _obstacles(layout, 0)
    random.seed(0)
    starts = sample_starts(N_ROBOTS, X_RANGE, Y_RANGE, layout, obs[:, :2], RADII)
    bx0, bx1 = layout.obstacle_band(X_RANGE)
    assert starts[directions == LEFT_TO_RIGHT, 0].max() < bx0
    assert starts[directions == RIGHT_TO_LEFT, 0].min() > bx1

    # ... and each half's goal band is the other half's start side.
    assert layout.goal_range_limits(X_RANGE, Y_RANGE, LEFT_TO_RIGHT)[0][0] > bx1
    assert layout.goal_range_limits(X_RANGE, Y_RANGE, RIGHT_TO_LEFT)[1][0] < bx0


def test_layout_is_reproducible_from_the_seeded_globals():
    """
    ``seed_episode`` seeds ``numpy.random`` and ``random``; drawing from those
    two globals is what lets a corridor episode be replayed by index.
    """
    layout = CorridorLayout()
    first = (_obstacles(layout, 7), None)
    random.seed(7)
    starts_a = sample_starts(N_ROBOTS, X_RANGE, Y_RANGE, layout, first[0][:, :2], RADII)

    obs_b = _obstacles(layout, 7)
    random.seed(7)
    starts_b = sample_starts(N_ROBOTS, X_RANGE, Y_RANGE, layout, obs_b[:, :2], RADII)

    np.testing.assert_array_equal(first[0], obs_b)
    np.testing.assert_array_equal(starts_a, starts_b)
