"""
Freeze the reference output of the 14-robot coarse rotation solve.

The golden file pins what ``compute_actions`` *observably produces* — the
per-robot rotation ``dθ`` recovered from the frames, the sub-step counts and
the translation command — for a fixed set of states × all 22 move-groups,
computed with ``nonlinear_solver="bfgs_fd"`` (the original scipy BFGS with
finite differences).

Why the observable output and not ``t*``: the objective is blind to the null
space of ``A_m``, so ``t*`` is not unique, but ``dθ = A_full @ t*`` is what the
simulator executes and what the shield sweeps.  Two solvers agreeing on the
objective while disagreeing on ``dθ`` would be a behaviour change — that is
exactly what this file is here to catch.

Regenerate only when the reference itself is meant to move (a deliberate
objective/tie-break change, or a scipy upgrade whose effect you have checked):

    python -m robot_nav.scripts.make_coarse_golden
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_nav.models.MARL.capswitcher_14.configs import (
    MOVE_GROUPS,
    N_ROBOTS,
    make_coarse_steering,
)

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / \
    "coarse_golden_14.npz"

# Fixed sampling of the reference states — a seed change invalidates the file.
N_STATES = 20
SEED = 20260813
MOVE_DISTANCE = 0.5
STEP_TIME = 0.3
ANG_MAX = 1.0
LIN_MAX = 0.5


def reference_states(n: int = N_STATES, seed: int = SEED) -> np.ndarray:
    """
    ``(n, N_ROBOTS, 11)`` robot-state blocks in the layout ``compute_actions``
    reads: 0,1 = px,py; 2,3 = cosθ,sinθ; 9,10 = gx,gy.  The untouched columns
    stay zero — the rotation solve never looks at them.
    """
    rng = np.random.default_rng(seed)
    states = np.zeros((n, N_ROBOTS, 11), dtype=np.float64)
    states[:, :, 0:2] = rng.uniform(-5.0, 5.0, size=(n, N_ROBOTS, 2))
    theta = rng.uniform(-np.pi, np.pi, size=(n, N_ROBOTS))
    states[:, :, 2] = np.cos(theta)
    states[:, :, 3] = np.sin(theta)
    states[:, :, 9:11] = rng.uniform(-5.0, 5.0, size=(n, N_ROBOTS, 2))
    return states


def frames_to_rotation(rotation_frames: list, step_time: float) -> np.ndarray:
    """Recover the executed per-robot ``dθ`` from the rotation frames."""
    if not rotation_frames:
        return np.zeros(N_ROBOTS, dtype=np.float64)
    ang = np.asarray(rotation_frames, dtype=np.float64)[:, :, 1]   # (n_rot, N)
    return ang.sum(axis=0) * step_time


def build(solver: str = "bfgs_fd", states: np.ndarray | None = None) -> dict:
    """Run every (state, group) through ``compute_actions`` and tabulate."""
    states = reference_states() if states is None else states
    cs = make_coarse_steering(
        move_distance=MOVE_DISTANCE, method="nonlinear", step_time=STEP_TIME,
        ang_max=ANG_MAX, lin_max=LIN_MAX, nonlinear_solver=solver,
    )
    n_s, n_g = states.shape[0], len(MOVE_GROUPS)
    out = {
        "d_theta":   np.zeros((n_s, n_g, N_ROBOTS)),
        "n_rot":     np.zeros((n_s, n_g), dtype=np.int64),
        "n_trans":   np.zeros((n_s, n_g), dtype=np.int64),
        "lin_cmd":   np.zeros((n_s, n_g)),
        "objective": np.zeros((n_s, n_g)),
        "iterations": np.zeros((n_s, n_g), dtype=np.int64),
    }
    for i, s in enumerate(states):
        for g in range(n_g):
            rot, trans = cs.compute_actions(s, g)
            out["d_theta"][i, g] = frames_to_rotation(rot, STEP_TIME)
            out["n_rot"][i, g] = len(rot)
            out["n_trans"][i, g] = len(trans)
            out["lin_cmd"][i, g] = trans[0][0][0] if trans else 0.0
            out["objective"][i, g] = objective_of(s, g, out["d_theta"][i, g])
            out["iterations"][i, g] = cs.last_solve_iterations
    return out


def objective_of(state: np.ndarray, group: int, d_theta: np.ndarray) -> float:
    """
    Members' total distance-to-goal reduction under ``d_theta`` — the quantity
    the solve maximises, evaluated straight from the executed rotation so it is
    solver-agnostic (negated: lower is better, matching ``neg_progress``).
    """
    s = np.asarray(state, dtype=np.float64)
    m = np.asarray(MOVE_GROUPS[group], dtype=int)
    p = np.stack([s[m, 0], s[m, 1]], axis=1)
    g = np.stack([s[m, 9], s[m, 10]], axis=1)
    theta = np.arctan2(s[m, 3], s[m, 2]) + d_theta[m]
    dist0 = np.linalg.norm(g - p, axis=1)
    p_new = p + MOVE_DISTANCE * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return float(-np.sum(dist0 - np.linalg.norm(g - p_new, axis=1)))


def main() -> None:
    data = build("bfgs_fd")
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOLDEN_PATH,
        states=reference_states(),
        seed=np.int64(SEED),
        move_distance=np.float64(MOVE_DISTANCE),
        step_time=np.float64(STEP_TIME),
        ang_max=np.float64(ANG_MAX),
        lin_max=np.float64(LIN_MAX),
        **data,
    )
    n = data["objective"].size
    print(
        f"Wrote {GOLDEN_PATH}\n"
        f"  {N_STATES} states × {len(MOVE_GROUPS)} groups = {n} solves\n"
        f"  mean objective {data['objective'].mean():.4f}, "
        f"mean BFGS iterations {data['iterations'].mean():.1f}\n"
        f"  rotation sub-steps: mean {data['n_rot'].mean():.1f}, "
        f"max {data['n_rot'].max()}"
    )


if __name__ == "__main__":
    main()
