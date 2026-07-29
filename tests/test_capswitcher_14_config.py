"""
Config tests for the 14-robot instantiation: the move-group lists are pinned to
the exact sets settled for the paper, and the actuation matrix has the expected
shape/rank with no artificial reduction.
"""

import numpy as np

from robot_nav.models.MARL.capswitcher_14.configs import (
    A_FULL,
    MOVE_GROUPS,
    N_MOVE_GROUPS,
    N_ROBOTS,
    make_coarse_steering,
)

EXPECTED_SIZE_3 = [
    [2, 6, 10], [4, 6, 12], [5, 6, 13], [8, 10, 12], [9, 10, 13], [11, 12, 13],
]
EXPECTED_SIZE_4 = [
    [0, 2, 4, 6], [0, 2, 8, 10], [0, 4, 8, 12],
    [1, 2, 5, 6], [1, 2, 9, 10], [1, 5, 9, 13],
    [3, 4, 5, 6], [3, 4, 11, 12], [3, 5, 11, 13],
    [7, 8, 9, 10], [7, 8, 11, 12], [7, 9, 11, 13],
]
EXPECTED_SIZE_7 = [
    [0, 2, 4, 6, 8, 10, 12],
    [1, 2, 5, 6, 9, 10, 13],
    [3, 4, 5, 6, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13],
]


def test_move_groups_are_exactly_the_settled_22():
    assert N_MOVE_GROUPS == 22
    by_size = {3: [], 4: [], 7: []}
    for g in MOVE_GROUPS:
        by_size[len(g)].append(g)
    assert sorted(by_size[3]) == EXPECTED_SIZE_3
    assert sorted(by_size[4]) == EXPECTED_SIZE_4
    assert sorted(by_size[7]) == EXPECTED_SIZE_7


def test_move_groups_sorted_by_size_then_indices():
    keys = [(len(g), g) for g in MOVE_GROUPS]
    assert keys == sorted(keys)


def test_actuation_matrix_shape_and_rank():
    assert A_FULL.shape == (N_ROBOTS, 5)
    # All-ones empty-group column last.
    assert np.all(A_FULL[:, -1] == 1.0)
    # Binary entries, full column rank 5 — naturally rank-deficient (5 < 14)
    # without any artificial column drop.
    assert set(np.unique(A_FULL)) <= {0.0, 1.0}
    assert np.linalg.matrix_rank(A_FULL) == 5


def test_actuation_columns_are_complements_of_original_groups():
    # Robot i has code i+1; original group j = robots with bit j set; the
    # actuation column j is its complement (non-members rotate).
    for j in range(4):
        member = np.array([((i + 1) >> j) & 1 for i in range(N_ROBOTS)])
        assert np.array_equal(A_FULL[:, j], 1.0 - member)


def _state_facing_east(n: int, goal_offset=(3.0, 0.0)) -> np.ndarray:
    """Simple (N, 11) state: robots on a line facing +x, goals offset."""
    s = np.zeros((n, 11), dtype=np.float64)
    s[:, 0] = np.arange(n) * 2.0          # px
    s[:, 1] = 0.0                          # py
    s[:, 2] = 1.0                          # cos θ
    s[:, 3] = 0.0                          # sin θ
    s[:, 9] = s[:, 0] + goal_offset[0]     # gx
    s[:, 10] = s[:, 1] + goal_offset[1]    # gy
    return s


def test_coarse_steering_deterministic_and_frame_shapes():
    cs = make_coarse_steering()
    state = _state_facing_east(N_ROBOTS, goal_offset=(2.0, 2.0))
    rot1, trans1 = cs.compute_actions(state, group=0)
    rot2, trans2 = cs.compute_actions(state, group=0)
    # No column drop, no RNG: identical output on repeat calls.
    assert rot1 == rot2 and trans1 == trans2
    for frame in rot1:
        assert len(frame) == N_ROBOTS
        assert all(pair[0] == 0.0 for pair in frame)          # rotation: no lin
    members = set(int(i) for i in cs.members_of(0))
    for frame in trans1:
        assert all(pair[1] == 0.0 for pair in frame)          # translation: no ang
        for i, pair in enumerate(frame):
            assert (pair[0] > 0.0) == (i in members)


def test_translation_realises_move_distance_for_members_only():
    cs = make_coarse_steering(move_distance=0.5)
    state = _state_facing_east(N_ROBOTS, goal_offset=(3.0, 0.0))
    group = 21  # a size-7 group
    _, trans = cs.compute_actions(state, group)
    total = np.zeros(N_ROBOTS)
    for frame in trans:
        total += np.array([pair[0] for pair in frame]) * cs.step_time
    members = cs.members_of(group)
    assert np.allclose(total[members], cs.move_distance)
    non_members = np.setdiff1d(np.arange(N_ROBOTS), members)
    assert np.allclose(total[non_members], 0.0)


def test_velocity_bounds_respected():
    cs = make_coarse_steering(move_distance=1.7)
    state = _state_facing_east(N_ROBOTS, goal_offset=(-3.0, 1.0))  # big rotations
    rot, trans = cs.compute_actions(state, group=6)  # a size-4 group
    for frame in rot:
        assert all(abs(pair[1]) <= cs.ang_max + 1e-12 for pair in frame)
    for frame in trans:
        assert all(pair[0] <= cs.lin_max + 1e-12 for pair in frame)


def test_selectable_groups_and_members_match_config():
    cs = make_coarse_steering()
    assert cs.selectable_groups() == list(range(22))
    for gid, expected in enumerate(MOVE_GROUPS):
        assert cs.members_of(gid).tolist() == expected
