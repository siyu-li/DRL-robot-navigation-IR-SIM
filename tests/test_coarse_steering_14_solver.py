"""
The 14-robot nonlinear rotation solve: analytic gradient, the default solver's
quality, and the bound on the rotation it is allowed to execute.

Context.  The objective scores only the *members'* progress and is periodic in
each member's heading, so the landscape is strongly multi-modal: which local
optimum a solver finds is a property of its path, not of the objective, and two
solvers can agree on the objective while disagreeing on ``dθ`` by whole turns.
``compute_actions`` therefore wraps ``dθ`` into (−π, π] — same final headings,
bounded ``n_rot`` — and this file pins the objective and that bound rather than
``dθ`` itself.

Self-contained by design: the reference states are generated from a fixed seed
and the ``bfgs_fd`` baseline is computed live, so there is no golden file to
keep in step with the code.  (There was one, ``tests/data/coarse_golden_14.npz``
plus ``robot_nav/scripts/make_coarse_golden.py``; both were removed on
2026-08-17 once the tests that compared ``dθ`` against it were dropped.)
"""

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher_14.configs import (
    A_FULL,
    MOVE_GROUPS,
    N_ROBOTS,
    make_coarse_steering,
)
from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
    CoarseSteering14,
)

# Fixed sampling of the reference states — changing the seed changes what every
# test in this file measures, so treat it as part of the fixture.
N_STATES = 20
SEED = 20260813
MOVE_DISTANCE = 0.5
STEP_TIME = 0.3
ANG_MAX = 1.0
LIN_MAX = 0.5


@pytest.fixture(scope="module")
def states() -> np.ndarray:
    """
    ``(N_STATES, N_ROBOTS, 11)`` robot-state blocks in the layout
    ``compute_actions`` reads: 0,1 = px,py; 2,3 = cosθ,sinθ; 9,10 = gx,gy.
    The untouched columns stay zero — the rotation solve never looks at them.
    """
    rng = np.random.default_rng(SEED)
    out = np.zeros((N_STATES, N_ROBOTS, 11), dtype=np.float64)
    out[:, :, 0:2] = rng.uniform(-5.0, 5.0, size=(N_STATES, N_ROBOTS, 2))
    theta = rng.uniform(-np.pi, np.pi, size=(N_STATES, N_ROBOTS))
    out[:, :, 2] = np.cos(theta)
    out[:, :, 3] = np.sin(theta)
    out[:, :, 9:11] = rng.uniform(-5.0, 5.0, size=(N_STATES, N_ROBOTS, 2))
    return out


def _steering(solver: str) -> CoarseSteering14:
    return make_coarse_steering(
        move_distance=MOVE_DISTANCE, method="nonlinear", step_time=STEP_TIME,
        ang_max=ANG_MAX, lin_max=LIN_MAX, nonlinear_solver=solver,
    )


def _rotation_of(rotation_frames: list) -> np.ndarray:
    """Recover the executed per-robot ``dθ`` from the rotation frames."""
    if not rotation_frames:
        return np.zeros(N_ROBOTS, dtype=np.float64)
    ang = np.asarray(rotation_frames, dtype=np.float64)[:, :, 1]   # (n_rot, N)
    return ang.sum(axis=0) * STEP_TIME


def _objective_of(state: np.ndarray, group: int, d_theta: np.ndarray) -> float:
    """
    Members' total distance-to-goal reduction under ``d_theta`` — the quantity
    the solve maximises, evaluated from the *executed* rotation so it is
    solver-agnostic (negated: lower is better).
    """
    s = np.asarray(state, dtype=np.float64)
    m = np.asarray(MOVE_GROUPS[group], dtype=int)
    p = np.stack([s[m, 0], s[m, 1]], axis=1)
    g = np.stack([s[m, 9], s[m, 10]], axis=1)
    theta = np.arctan2(s[m, 3], s[m, 2]) + d_theta[m]
    dist0 = np.linalg.norm(g - p, axis=1)
    p_new = p + MOVE_DISTANCE * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return float(-np.sum(dist0 - np.linalg.norm(g - p_new, axis=1)))


def _mean_objective(solver: str, states: np.ndarray) -> float:
    """Mean objective over every (state, group) pair for ``solver``."""
    cs = _steering(solver)
    return float(np.mean([
        _objective_of(s, g, _rotation_of(cs.compute_actions(s, g)[0]))
        for s in states for g in range(len(MOVE_GROUPS))
    ]))


def _terms_at(state, group, t):
    """``(f, grad)`` of the rotation objective at ``t``."""
    s = np.asarray(state, dtype=np.float64)
    m = np.asarray(MOVE_GROUPS[group], dtype=int)
    rel = np.stack([s[m, 9] - s[m, 0], s[m, 10] - s[m, 1]], axis=1)
    return CoarseSteering14._rotation_terms(
        t,
        np.arctan2(s[m, 3], s[m, 2]),
        A_FULL[m, :],
        np.linalg.norm(rel, axis=1),
        np.arctan2(rel[:, 1], rel[:, 0]),
        MOVE_DISTANCE,
    )


# ---------------------------------------------------------------------------
# The analytic gradient — the whole speedup rests on it being right
# ---------------------------------------------------------------------------

def test_gradient_matches_finite_differences(states):
    rng = np.random.default_rng(7)
    for state in states[:5]:
        for group in (0, 7, 21):
            t = rng.normal(size=A_FULL.shape[1])
            _, grad = _terms_at(state, group, t)
            eps = 1e-6
            fd = np.empty_like(grad)
            for i in range(t.size):
                step = np.zeros_like(t)
                step[i] = eps
                f_plus, _ = _terms_at(state, group, t + step)
                f_minus, _ = _terms_at(state, group, t - step)
                fd[i] = (f_plus - f_minus) / (2 * eps)
            assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# The default solver
# ---------------------------------------------------------------------------

def test_default_solver_is_the_lean_one():
    cs = make_coarse_steering(method="nonlinear")
    assert cs.nonlinear_solver == "bfgs_lean"


def test_lean_bfgs_is_quality_neutral(states):
    """
    ``bfgs_lean`` uses a different line search from the scipy reference, so on
    this multi-modal landscape it lands on a different — but equally good —
    local optimum in most solves.  The objective is the thing that has to hold;
    ``dθ`` itself is not comparable across solvers here.  Measured over 6600
    random solves the gap is 0.02%; the 2% bar leaves room for the 440 solves
    this fixture affords.
    """
    reference = _mean_objective("bfgs_fd", states)
    lean = _mean_objective("bfgs_lean", states)
    assert abs(lean - reference) < 0.02 * abs(reference)


# ---------------------------------------------------------------------------
# The rotation is bounded, whatever the solver returns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["bfgs_lean", "bfgs", "bfgs_fd"])
def test_executed_rotation_never_exceeds_half_a_turn(states, solver):
    """
    ``compute_actions`` wraps ``dθ`` into (−π, π], so ``n_rot`` is bounded by
    ``ceil(π / (ang_max·step_time))`` no matter which local optimum the solver
    walked to.  Without this the lean solver reached 74610 rad — 248699 frames
    for one coarse edge — and OOM-killed 14-robot runs.
    """
    cs = _steering(solver)
    cap = int(np.ceil(np.pi / (cs.ang_max * cs.step_time)))
    for state in states[:5]:
        for group in range(len(MOVE_GROUPS)):
            rot, _ = cs.compute_actions(state, group)
            assert len(rot) <= cap
            assert np.abs(_rotation_of(rot)).max() <= np.pi + 1e-9
