"""
Sanity-check the dynamics-aware features of the value/prior redesign
(``capswitcher_14/docs/value_prior_redesign.md`` §4): does the expansion-free
rotation preview

    dtheta_hat = wrap(P @ dtheta_desired),      P = A @ pinv(A)

capture what one coarse control of ``CoarseSteering14`` actually does?

Three claims are tested against the executed primitive (sim-free — the
primitive's own frames are ground truth, exactly what ``ForwardModel14``
integrates):

1. **Exactness under least_squares.**  The LS primitive applies
   ``wrap(A @ pinv(A) @ dtheta_desired)`` by construction, so the preview must
   match the executed per-robot rotation to numerical precision — and so must
   everything derived from it: post-rotation member heading error, bystander
   churn, member progress, and the residual feature
   ``|wrap((I - P) dtheta_desired)|`` vs the actual leftover heading error.

2. **First-order fidelity under nonlinear.**  The BFGS solve starts at the LS
   solution and only improves member progress, so the preview should still
   *rank* the 22 move-groups by progress essentially as well as materializing
   them (per-state Spearman), and per-robot rotations should correlate
   strongly.  This is the property the prior needs — it ranks edges, it does
   not integrate them.

3. **The preview is informative, not trivial.**  A dynamics-blind preview
   (assume every member turns perfectly to its goal — what the old features
   implicitly assumed) collapses to ~``|members| * move_distance`` and should
   rank groups clearly worse than the P-aware preview.

Usage:
    PYTHONPATH=. python -m robot_nav.check_coupling_features --states 30 [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from robot_nav.models.MARL.capswitcher_14.configs import (
    A_FULL,
    MOVE_GROUPS,
    N_ROBOTS,
    make_coarse_steering,
)

P = A_FULL @ np.linalg.pinv(A_FULL)          # (N, N) coupling projection


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def sample_state(rng: np.random.Generator) -> np.ndarray:
    """Random (N, 11) robot_state: poses/goals in an 8 m box, dist >= 0.5."""
    px = rng.uniform(1.0, 9.0, N_ROBOTS)
    py = rng.uniform(1.0, 9.0, N_ROBOTS)
    theta = rng.uniform(-np.pi, np.pi, N_ROBOTS)
    while True:
        gx = rng.uniform(1.0, 9.0, N_ROBOTS)
        gy = rng.uniform(1.0, 9.0, N_ROBOTS)
        if np.all(np.hypot(gx - px, gy - py) >= 0.5):
            break
    s = np.zeros((N_ROBOTS, 11))
    s[:, 0], s[:, 1] = px, py
    s[:, 2], s[:, 3] = np.cos(theta), np.sin(theta)
    s[:, 9], s[:, 10] = gx, gy
    return s


def preview(s: np.ndarray, members: np.ndarray, move: float,
            target: str = "joint") -> dict:
    """
    The §4.3 rotation-preview features — one small solve, no transition.

    target="joint":   d_hat = wrap(P @ d_des) — mirrors the least_squares
                      primitive (exact for it).
    target="members": d_hat = wrap(A @ pinv(A_members) @ d_des[members]) —
                      steer the members' headings as well as the actuation
                      allows, ignoring bystanders; mirrors what the nonlinear
                      progress solve effectively optimises (first-order), and
                      is the same solve precise groups use (pinv(A_S)).
    """
    px, py, gx, gy = s[:, 0], s[:, 1], s[:, 9], s[:, 10]
    theta = np.arctan2(s[:, 3], s[:, 2])
    d_des = _wrap(np.arctan2(gy - py, gx - px) - theta)      # desired turns
    if target == "joint":
        d_hat = _wrap(P @ d_des)                              # previewed turns
    else:
        t = np.linalg.pinv(A_FULL[members, :]) @ d_des[members]
        d_hat = _wrap(A_FULL @ t)
    residual = _wrap(d_des - d_hat)                           # (I-P) d_des, wrapped
    theta_new = theta + d_hat
    dist0 = np.hypot(gx - px, gy - py)
    nx = px + move * np.cos(theta_new)
    ny = py + move * np.sin(theta_new)
    progress = float(np.sum(dist0[members] - np.hypot(gx - nx, gy - ny)[members]))
    bystanders = np.setdiff1d(np.arange(N_ROBOTS), members)
    return {
        "d_hat": d_hat,
        "residual_abs": np.abs(residual),
        "member_err_after": np.abs(_wrap(d_des - d_hat))[members],
        "bystander_churn": np.abs(d_hat)[bystanders],
        "progress": progress,
        # dynamics-blind baseline: pretend every member turns fully to goal
        "progress_naive": float(
            np.sum(dist0[members] - np.abs(dist0[members] - move))
        ),
    }


def executed(coarse, s: np.ndarray, group: int) -> dict:
    """Ground truth from the primitive's own frames (what the sim integrates)."""
    rot, _trans = coarse.compute_actions(s, group)
    d_act = np.zeros(N_ROBOTS)
    for frame in rot:
        d_act += np.asarray(frame)[:, 1] * coarse.step_time
    d_act = _wrap(d_act)
    px, py, gx, gy = s[:, 0], s[:, 1], s[:, 9], s[:, 10]
    theta = np.arctan2(s[:, 3], s[:, 2])
    d_des = _wrap(np.arctan2(gy - py, gx - px) - theta)
    members = coarse.members_of(group)
    move = coarse.move_distances[group]
    theta_new = theta + d_act
    dist0 = np.hypot(gx - px, gy - py)
    nx = px + move * np.cos(theta_new)
    ny = py + move * np.sin(theta_new)
    progress = float(np.sum(dist0[members] - np.hypot(gx - nx, gy - ny)[members]))
    bystanders = np.setdiff1d(np.arange(N_ROBOTS), members)
    return {
        "d_act": d_act,
        "member_err_after": np.abs(_wrap(d_des - d_act))[members],
        "bystander_churn": np.abs(d_act)[bystanders],
        "progress": progress,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move-distance", type=float, default=0.5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    states = [sample_state(rng) for _ in range(args.states)]
    n_groups = len(MOVE_GROUPS)

    # ---- 1. exactness under least_squares --------------------------------
    ls = make_coarse_steering(move_distance=args.move_distance,
                              method="least_squares")
    worst_rot = worst_prog = worst_err = worst_res = 0.0
    for s in states:
        for g in range(n_groups):
            pv = preview(s, ls.members_of(g), args.move_distance)
            ex = executed(ls, s, g)
            worst_rot = max(worst_rot, float(np.max(np.abs(_wrap(pv["d_hat"] - ex["d_act"])))))
            worst_prog = max(worst_prog, abs(pv["progress"] - ex["progress"]))
            worst_err = max(worst_err, float(np.max(np.abs(pv["member_err_after"] - ex["member_err_after"]))))
            # residual feature == actual leftover heading error (all robots)
            px, py, gx, gy = s[:, 0], s[:, 1], s[:, 9], s[:, 10]
            theta = np.arctan2(s[:, 3], s[:, 2])
            d_des = _wrap(np.arctan2(gy - py, gx - px) - theta)
            left = np.abs(_wrap(d_des - ex["d_act"]))
            worst_res = max(worst_res, float(np.max(np.abs(pv["residual_abs"] - left))))
    tol = 1e-9
    ok = max(worst_rot, worst_prog, worst_err, worst_res) < tol
    print("== 1. least_squares: preview must equal execution ==")
    print(f"  max |preview - executed|  rotation: {worst_rot:.2e}   progress: {worst_prog:.2e}")
    print(f"                            member-err-after: {worst_err:.2e}   residual-vs-leftover: {worst_res:.2e}")
    print(f"  {'PASS' if ok else 'FAIL'} (tol {tol:.0e})\n")

    # ---- 2. first-order fidelity under nonlinear -------------------------
    nl = make_coarse_steering(move_distance=args.move_distance, method="nonlinear")
    stats = {k: {"rank": [], "rot": [], "churn": [], "mae": []}
             for k in ("joint", "members")}
    naive_rank = []
    for s in states:
        prog = {"joint": [], "members": [], "naive": [], "ex": []}
        for g in range(n_groups):
            ex = executed(nl, s, g)
            prog["ex"].append(ex["progress"])
            for k in ("joint", "members"):
                pv = preview(s, nl.members_of(g), args.move_distance, target=k)
                prog[k].append(pv["progress"])
                stats[k]["rot"].append(np.corrcoef(pv["d_hat"], ex["d_act"])[0, 1])
                stats[k]["churn"].append(
                    np.corrcoef(pv["bystander_churn"], ex["bystander_churn"])[0, 1])
                stats[k]["mae"].append(abs(pv["progress"] - ex["progress"]))
                if k == "joint":
                    prog["naive"].append(pv["progress_naive"])
        for k in ("joint", "members"):
            stats[k]["rank"].append(spearmanr(prog[k], prog["ex"]).statistic)
        naive_rank.append(spearmanr(prog["naive"], prog["ex"]).statistic)

    print("== 2. nonlinear primitive: which preview is the right first-order model? ==")
    print(f"  {'preview':<28}{'rank Spearman':>16}{'rot Pearson':>14}"
          f"{'churn Pearson':>15}{'prog MAE (m)':>14}")
    for k, label in (("joint", "joint-LS  P @ d_des"),
                     ("members", "member-targeted pinv(A_m)")):
        st = stats[k]
        print(f"  {label:<28}{np.mean(st['rank']):>10.3f} (min "
              f"{np.min(st['rank']):>5.2f}){np.mean(st['rot']):>10.3f}"
              f"{np.mean(st['churn']):>15.3f}{np.mean(st['mae']):>14.3f}")
    print(f"  {'dynamics-blind (full turn)':<28}{np.mean(naive_rank):>10.3f} (min "
          f"{np.min(naive_rank):>5.2f})")
    print(f"  executed progress mean: {np.mean(np.abs(prog['ex'])):.3f} m\n")

    # ---- 3. within-size ranking: the discrimination the prior needs ------
    # Across sizes, executed nonlinear progress is dominated by |members| *
    # move (the solver nearly achieves full member progress), which the naive
    # preview encodes perfectly.  The prior's real job is picking WHICH
    # size-4 group — where naive is constant and carries zero information.
    sizes = np.array([len(g) for g in MOVE_GROUPS])
    ratio = []
    within = {k: {sz: [] for sz in np.unique(sizes)} for k in ("joint", "members")}
    for s in states:
        prog = {"joint": [], "members": [], "ex": []}
        for g in range(n_groups):
            ex = executed(nl, s, g)
            prog["ex"].append(ex["progress"])
            ratio.append(ex["progress"] /
                         (sizes[g] * args.move_distance))
            for k in ("joint", "members"):
                prog[k].append(
                    preview(s, nl.members_of(g), args.move_distance, target=k)["progress"])
        for sz in np.unique(sizes):
            m = sizes == sz
            for k in ("joint", "members"):
                r = spearmanr(np.asarray(prog[k])[m], np.asarray(prog["ex"])[m]).statistic
                if np.isfinite(r):
                    within[k][sz].append(r)
    print("== 3. within-size-class ranking (naive preview = constant here) ==")
    print(f"  executed / ideal member progress: mean "
          f"{np.mean(ratio):.3f} (how close the solver gets to full progress)")
    for sz in np.unique(sizes):
        row = "  ".join(
            f"{k}: {np.mean(within[k][sz]):.3f}" for k in ("joint", "members"))
        print(f"  size-{sz} groups ({np.sum(sizes == sz):2d} per state): Spearman {row}")
    ok2 = all(np.mean(within["members"][sz]) > 0.3 for sz in np.unique(sizes))
    if ok2:
        print("  linear member-targeted preview ranks within every size class "
              "-> tier-1 preview is sufficient for the nonlinear primitive")
    else:
        print("  linear previews cannot rank within every size class -> use the "
              "exact-solve preview (tier 2, design doc §4.3); cost measured below")

    # ---- 4. cost of the exact preview (tier 2) ----------------------------
    # The nonlinear solve is path-dependent (multi-modal objective, local
    # BFGS), so no linear formula reproduces it.  But the solve itself —
    # without frames materialisation charges, without the swept-clearance vet
    # — is cheap and expansion-free.  Measure what "run the real rotation
    # solve per group token" would cost the prior per node.
    import time
    t0 = time.perf_counter()
    n_solves = 0
    for s in states[:10]:
        for g in range(n_groups):
            nl.compute_actions(s, g)
            n_solves += 1
    per_solve = (time.perf_counter() - t0) / n_solves
    print("\n== 4. exact-preview cost (rotation solve, no sweep/vet) ==")
    print(f"  mean per group: {per_solve * 1e3:.2f} ms  ->  per node "
          f"({n_groups} groups): {per_solve * n_groups * 1e3:.1f} ms")
    print("  (exact by construction for the deployed primitive; use this for "
          "the action-token features when method='nonlinear')")


if __name__ == "__main__":
    main()
