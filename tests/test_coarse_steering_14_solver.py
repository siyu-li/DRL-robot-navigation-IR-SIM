"""
The 14-robot nonlinear rotation solve: derivatives, the golden reference, and
the null-space tie-break.

Context — why this file is more than a smoke test.  The objective scores only
the *members'* progress, and ``rank(A_m)`` is 3–4 against ``K = 5``, so:

1. the optimum is a whole subspace, and which point in it you land on decides
   how far the **non-members** are turned — which feeds the swept path, the
   shield's clearance verdict and ``n_rot``;
2. the landscape is strongly multi-modal (the reference solver turns some robot
   past π in half of all solves), so *which local optimum* a solver finds is a
   property of its path, not of the objective.

Both make "same objective value" far too weak a check for a solver swap.  The
golden file pins the executed rotation ``dθ`` itself.
"""

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher_14.configs import (
    A_FULL,
    MOVE_GROUPS,
    make_coarse_steering,
)
from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
    CoarseSteering14,
)
from robot_nav.scripts.make_coarse_golden import (
    GOLDEN_PATH,
    MOVE_DISTANCE,
    STEP_TIME,
    build,
    frames_to_rotation,
    objective_of,
)

PINV_A = np.linalg.pinv(A_FULL)


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip(
            f"{GOLDEN_PATH} missing — regenerate with "
            "`python -m robot_nav.scripts.make_coarse_golden`"
        )
    return dict(np.load(GOLDEN_PATH))


def _terms_at(state, group, t):
    """``(f, grad, hess)`` of the rotation objective at ``t``."""
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
        want_hess=True,
    )


# ---------------------------------------------------------------------------
# The analytic derivatives — the whole speedup rests on these being right
# ---------------------------------------------------------------------------

def test_gradient_matches_finite_differences(golden):
    rng = np.random.default_rng(7)
    for state in golden["states"][:5]:
        for group in (0, 7, 21):
            t = rng.normal(size=A_FULL.shape[1])
            _, grad, _ = _terms_at(state, group, t)
            eps = 1e-6
            fd = np.empty_like(grad)
            for i in range(t.size):
                step = np.zeros_like(t)
                step[i] = eps
                f_plus, _, _ = _terms_at(state, group, t + step)
                f_minus, _, _ = _terms_at(state, group, t - step)
                fd[i] = (f_plus - f_minus) / (2 * eps)
            assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_hessian_matches_finite_differences(golden):
    rng = np.random.default_rng(8)
    for state in golden["states"][:3]:
        for group in (0, 11, 21):
            t = rng.normal(size=A_FULL.shape[1])
            _, _, hess = _terms_at(state, group, t)
            eps = 1e-6
            fd = np.empty_like(hess)
            for i in range(t.size):
                step = np.zeros_like(t)
                step[i] = eps
                _, g_plus, _ = _terms_at(state, group, t + step)
                _, g_minus, _ = _terms_at(state, group, t - step)
                fd[:, i] = (g_plus - g_minus) / (2 * eps)
            assert np.allclose(hess, fd, rtol=1e-4, atol=1e-6)
            assert np.allclose(hess, hess.T, atol=1e-12)


# ---------------------------------------------------------------------------
# The golden reference
# ---------------------------------------------------------------------------

def test_reference_solver_reproduces_golden(golden):
    """``bfgs_fd`` is the frozen reference: any drift here is scipy's, not ours."""
    out = build("bfgs_fd", states=golden["states"])
    assert np.allclose(out["d_theta"], golden["d_theta"], rtol=0, atol=1e-12)
    assert np.array_equal(out["n_rot"], golden["n_rot"])
    assert np.array_equal(out["n_trans"], golden["n_trans"])


def test_analytic_gradient_solver_matches_the_reference(golden):
    """
    The default ``bfgs`` only replaces finite differences with the exact
    gradient, so it must reproduce the reference's *rotation*, not merely its
    objective value.  Measured: max |Δdθ| 1.8e-5 rad, objective within 4e-12.
    """
    out = build("bfgs", states=golden["states"])
    assert np.abs(out["d_theta"] - golden["d_theta"]).max() < 1e-3
    assert np.abs(out["objective"] - golden["objective"]).max() < 1e-8
    # Sub-step counts are what the simulator actually executes.
    assert np.array_equal(out["n_rot"], golden["n_rot"])
    assert np.array_equal(out["n_trans"], golden["n_trans"])


def test_default_solver_is_the_faithful_one():
    cs = make_coarse_steering(method="nonlinear")
    assert cs.nonlinear_solver == "bfgs"


# ---------------------------------------------------------------------------
# The null-space tie-break
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["bfgs", "bfgs_fd", "bfgs_lean", "newton"])
def test_solvers_preserve_the_null_space_tiebreak(golden, solver):
    """
    Every solver must leave ``t``'s component in ``null(A_m)`` exactly where
    ``t_ls`` put it.  That component is invisible to the objective but rotates
    the non-members, so a solver that moved it would silently change the swept
    path — a behaviour change wearing an optimisation's clothes.  Steps built
    from ``∇f = A_mᵀw`` (and from ``(H + λI)⁻¹∇f``) cannot move it.
    """
    from scipy.linalg import null_space

    cs = make_coarse_steering(
        move_distance=MOVE_DISTANCE, method="nonlinear", step_time=STEP_TIME,
        nonlinear_solver=solver,
    )
    for state in golden["states"][:5]:
        s = np.asarray(state, dtype=np.float64)
        theta = np.arctan2(s[:, 3], s[:, 2])
        phi = np.arctan2(s[:, 10] - s[:, 1], s[:, 9] - s[:, 0])
        d_desired = (phi - theta + np.pi) % (2 * np.pi) - np.pi
        t_ls = PINV_A @ d_desired
        for group in range(len(MOVE_GROUPS)):
            rot, _ = cs.compute_actions(s, group)
            # A_FULL has full column rank, so t* is recoverable from dθ.
            t_star = PINV_A @ frames_to_rotation(rot, STEP_TIME)
            Z = null_space(A_FULL[np.asarray(MOVE_GROUPS[group]), :])
            assert Z.shape[1] > 0, "every move-group should be rank-deficient"
            assert np.allclose(Z.T @ t_star, Z.T @ t_ls, atol=1e-6)


# ---------------------------------------------------------------------------
# The fast solvers: documented deviations, so nobody adopts one by accident
# ---------------------------------------------------------------------------

def test_lean_bfgs_is_quality_neutral_but_not_identical(golden):
    """
    ``bfgs_lean`` uses a different line search, so on this multi-modal
    landscape it lands on a different — but equally good — local optimum in
    most solves.  Quality-neutral on the mean; adopt only behind an end-to-end
    A/B, never on the strength of a unit test.
    """
    out = build("bfgs_lean", states=golden["states"])
    delta = out["objective"].mean() - golden["objective"].mean()
    assert abs(delta) < 0.02 * abs(golden["objective"].mean())   # ≤2% of mean
    assert np.abs(out["d_theta"] - golden["d_theta"]).max() > 1e-3  # not identical


def test_newton_finds_worse_optima(golden):
    """
    Pins the reason ``newton`` is not the default: it converges to a nearby
    *worse* stationary point (~11% less member progress on average).  If a
    future change to the objective makes the landscape well-behaved, this test
    is the one that should start failing.
    """
    out = build("newton", states=golden["states"])
    ref = golden["objective"].mean()
    assert out["objective"].mean() > ref                 # worse (less negative)
    assert out["objective"].mean() < 0.75 * ref          # but far better than t_ls
