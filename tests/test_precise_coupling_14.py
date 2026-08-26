"""Unit tests for the coupled-rotation solver (physics fix, redesign §2)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher_14.configs import A_FULL, N_ROBOTS
from robot_nav.models.MARL.capswitcher_14.policies.precise_coupling import (
    PreciseCoupling,
)


@pytest.fixture()
def coupling() -> PreciseCoupling:
    return PreciseCoupling(A_FULL, ang_max=1.0)


def test_singleton_target_exact(coupling: PreciseCoupling) -> None:
    """Every single driven robot receives exactly its commanded rotation."""
    for r in range(N_ROBOTS):
        w = coupling.coupled_ang([r], [0.7])
        assert w[r] == pytest.approx(0.7, abs=1e-12)


def test_pair_targets_exact(coupling: PreciseCoupling) -> None:
    """Every pair has rank-2 rows: both commands are met exactly (pre-clip)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        i, j = rng.choice(N_ROBOTS, size=2, replace=False)
        cmd = rng.uniform(-0.5, 0.5, size=2)  # small: no bystander clipping
        w = coupling.coupled_ang([int(i), int(j)], cmd)
        assert w[i] == pytest.approx(cmd[0], abs=1e-9)
        assert w[j] == pytest.approx(cmd[1], abs=1e-9)


def test_bystanders_rotate(coupling: PreciseCoupling) -> None:
    """The fix's whole point: bystanders receive nonzero side-rotation."""
    w = coupling.coupled_ang([0], [1.0])
    bystanders = np.delete(np.abs(w), 0)
    assert bystanders.max() > 0.1  # measured ~0.75 on the canonical matrix
    # And it matches the analytic map A @ pinv(A_row0).
    expected = A_FULL @ np.linalg.pinv(A_FULL[[0], :]) @ np.array([1.0])
    np.testing.assert_allclose(w, np.clip(expected, -1.0, 1.0), atol=1e-12)


def test_ang_max_clipping() -> None:
    """Commands are clipped at the simulator cap, driven robots included."""
    c = PreciseCoupling(A_FULL, ang_max=0.5)
    w = c.coupled_ang([0], [1.0])          # commanded above the cap
    assert np.all(np.abs(w) <= 0.5 + 1e-12)
    assert w[0] == pytest.approx(0.5)


def test_zero_command_is_zero(coupling: PreciseCoupling) -> None:
    w = coupling.coupled_ang([3, 7], [0.0, 0.0])
    np.testing.assert_allclose(w, np.zeros(N_ROBOTS))


def test_cache_reuse_and_shape_check(coupling: PreciseCoupling) -> None:
    coupling.coupled_ang([1, 2], [0.1, 0.2])
    assert (1, 2) in coupling._C
    with pytest.raises(ValueError):
        coupling.coupled_ang([1, 2], [0.1])   # wrong command length
