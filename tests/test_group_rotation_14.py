"""Unit tests for uniform block rotation (group-based physics fix)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher_14.configs import A_FULL, N_ROBOTS
from robot_nav.models.MARL.capswitcher_14.policies.group_rotation import (
    GroupRotation,
)


@pytest.fixture()
def rotation() -> GroupRotation:
    return GroupRotation(A_FULL, ang_max=1.0)


def test_blocks_partition_and_are_realisable(rotation: GroupRotation) -> None:
    """The two bit-0 blocks partition the 14 robots and lie in col(A)."""
    blocks = {tuple(rotation.block_of(r)) for r in range(N_ROBOTS)}
    assert len(blocks) == 2
    a, b = sorted(blocks)
    assert sorted(a + b) == list(range(N_ROBOTS))
    assert len(a) == len(b) == 7
    proj = A_FULL @ np.linalg.pinv(A_FULL)
    for blk in (a, b):
        ind = np.zeros(N_ROBOTS)
        ind[list(blk)] = 1.0
        np.testing.assert_allclose(proj @ ind, ind, atol=1e-9)


def test_driven_robot_and_block_exact(rotation: GroupRotation) -> None:
    """The whole block turns at exactly the commanded rate; the rest hold."""
    for r in range(N_ROBOTS):
        w = rotation.coupled_ang([r], [0.7])
        block = rotation.block_of(r)
        np.testing.assert_allclose(w[block], 0.7, atol=1e-12)
        others = np.setdiff1d(np.arange(N_ROBOTS), block)
        np.testing.assert_allclose(w[others], 0.0, atol=1e-12)


def test_result_is_actuation_realisable(rotation: GroupRotation) -> None:
    """Every returned pattern lies in col(A) — it is a physical rotation."""
    proj = A_FULL @ np.linalg.pinv(A_FULL)
    rng = np.random.default_rng(0)
    for _ in range(20):
        i, j = rng.choice(N_ROBOTS, size=2, replace=False)
        cmd = rng.uniform(-0.5, 0.5, size=2)
        w = rotation.coupled_ang([int(i), int(j)], cmd)
        np.testing.assert_allclose(proj @ w, w, atol=1e-9)


def test_canonical_pairs_hit_both_blocks_exactly(rotation: GroupRotation) -> None:
    """Pairs [[0,1],[2,3],...] straddle the blocks: both commands are met."""
    for i in range(0, N_ROBOTS - 1, 2):
        w = rotation.coupled_ang([i, i + 1], [0.3, -0.4])
        assert w[i] == pytest.approx(0.3, abs=1e-12)
        assert w[i + 1] == pytest.approx(-0.4, abs=1e-12)


def test_same_block_commands_sum(rotation: GroupRotation) -> None:
    """Two driven robots in one block add their commands over that block."""
    r0 = 0
    block = rotation.block_of(r0)
    r1 = int(block[1])
    w = rotation.coupled_ang([r0, r1], [0.2, 0.3])
    np.testing.assert_allclose(w[block], 0.5, atol=1e-12)


def test_ang_max_clipping() -> None:
    rot = GroupRotation(A_FULL, ang_max=0.5)
    w = rot.coupled_ang([0], [1.0])
    assert np.all(np.abs(w) <= 0.5 + 1e-12)
    assert w[0] == pytest.approx(0.5)


def test_zero_command_is_zero(rotation: GroupRotation) -> None:
    w = rotation.coupled_ang([3, 7], [0.0, 0.0])
    np.testing.assert_allclose(w, np.zeros(N_ROBOTS))


def test_shape_check(rotation: GroupRotation) -> None:
    with pytest.raises(ValueError):
        rotation.coupled_ang([1, 2], [0.1])


def test_alternative_bits_all_realisable() -> None:
    """Any of the 4 code bits anchors a valid 7+7 partition."""
    for bit in range(4):
        rot = GroupRotation(A_FULL, ang_max=1.0, bit=bit)
        sizes = {len(rot.block_of(r)) for r in range(N_ROBOTS)}
        assert sizes == {7}
