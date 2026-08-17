"""
Analyse leaf cost-to-go shards — the decision point of the learned-leaf plan.

Sim-free: runs locally on shards written by ``collect_leaf_data.py``.

The question, in order of how much work the answer implies:

1. **How biased is the analytic heuristic?**  Ratio ``G / h`` summary.  The
   heuristic prices completion at the precise rate and is blind to coarse moves
   being 4–60× cheaper per robot-metre, so it is expected to overestimate.

2. **Is a corrected constant enough?**  Fit the single best scalar ``c`` in
   ``G ≈ c · h`` (in log space, so it is the geometric-mean ratio) and report
   the spread left over.  **If the residual spread is small, stop — retune α
   and skip the network entirely.**  This is the cheapest possible outcome and
   the reason this script runs before stage 2.

3. **If not, is what is left predictable?**  Report how much of the residual
   variance is explained by the two things a network would have to beat: the
   number of unreached robots, and the coarse/precise mix the policy actually
   used.  A residual that is mostly noise is not worth a network either.

Reported spreads are in **log space**, because the target is multiplicative:
``σ_log = 0.3`` means a typical prediction is off by a factor of ~1.35.

Usage:
    python -m robot_nav.analyze_leaf_data --data data/leaf_14e
    python -m robot_nav.analyze_leaf_data --data data/leaf_14e data/leaf_14l
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# A network is worth building only if a constant leaves this much log-spread.
# exp(0.35) ~ 1.42x typical error — below that, retuning alpha is competitive
# with anything a small net will reliably deliver.
_CONSTANT_IS_ENOUGH = 0.35


def load_shards(paths: list[str]) -> dict[str, np.ndarray]:
    """Concatenate every shard under ``paths`` (files or directories)."""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        files.extend(sorted(path.rglob("*.npz")) if path.is_dir() else [path])
    if not files:
        raise FileNotFoundError(f"no .npz shards under {paths}")

    keys = ("h", "G", "ratio", "n_unreached", "sum_dist", "mode", "group",
            "valid", "episode", "t")
    out: dict[str, list] = {k: [] for k in keys}
    scalars: dict[str, float] = {}
    policies: list[str] = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            for k in keys:
                out[k].append(z[k])
            for k in ("alpha", "goal_threshold", "precise_unit"):
                if k in z:
                    scalars[k] = float(z[k])
            policies.append(str(z["policy"]) if "policy" in z else f"{f.name} (?)")
    merged = {k: np.concatenate(v) for k, v in out.items()}
    merged.update({k: np.float64(v) for k, v in scalars.items()})
    merged["n_shards"] = len(files)
    merged["policies"] = sorted(set(policies))
    return merged


def _log_spread(x: np.ndarray) -> tuple[float, float]:
    """Robust (median, sigma) of a log-space sample; sigma from the IQR."""
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    return med, float((q75 - q25) / 1.349)


def _explained(log_ratio: np.ndarray, groups: np.ndarray) -> float:
    """
    Fraction of log-ratio variance explained by a grouping (one mean per
    group) — a model-free ceiling on what a feature can contribute.
    """
    total = float(np.var(log_ratio))
    if total <= 0.0:
        return 0.0
    resid = log_ratio.copy()
    for g in np.unique(groups):
        mask = groups == g
        resid[mask] -= log_ratio[mask].mean()
    return float(1.0 - np.var(resid) / total)


def report(d: dict[str, np.ndarray]) -> None:
    valid = d["valid"] & np.isfinite(d["ratio"]) & (d["ratio"] > 0.0)
    n_total, n = valid.size, int(valid.sum())
    if n == 0:
        raise SystemExit("no valid labelled decisions in these shards")

    ratio = d["ratio"][valid]
    log_ratio = np.log(ratio)
    n_unreached = d["n_unreached"][valid]
    mode = d["mode"][valid]

    print("=" * 72)
    print(f"Leaf heuristic bias — {n} labelled decisions "
          f"({n_total - n} censored/dropped), {d['n_shards']} shard(s)")
    print(f"alpha = {float(d.get('alpha', np.nan)):.1f} cost per robot-metre")
    policies = d.get("policies") or []
    for p in policies:
        print(f"  policy: {p}")
    if len(policies) > 1:
        print("  WARNING: shards from different policies are pooled into one "
              "fit.\n           A return is only valid for the policy that "
              "produced it — split\n           the runs unless you meant to "
              "average across them.")
    print("=" * 72)

    # -- 1. how biased ----------------------------------------------------
    med, sigma = _log_spread(log_ratio)
    print("\n1. Ratio  G / h_analytic")
    print(f"   median      {np.exp(med):.4f}   "
          f"(heuristic overestimates {1.0 / np.exp(med):.1f}x)")
    print(f"   IQR         [{np.percentile(ratio, 25):.4f}, "
          f"{np.percentile(ratio, 75):.4f}]")
    print(f"   p5 / p95    {np.percentile(ratio, 5):.4f} / "
          f"{np.percentile(ratio, 95):.4f}")

    # -- 2. is a constant enough? -----------------------------------------
    # Best scalar in log space = geometric mean ratio; residual spread is what
    # any learned correction would have to beat.
    c = float(np.exp(np.mean(log_ratio)))
    resid = log_ratio - np.log(c)
    resid_sigma = float(np.std(resid))
    print("\n2. Best constant correction  G ~ c * h_analytic")
    print(f"   c           {c:.4f}   (equivalently alpha -> "
          f"{float(d.get('alpha', np.nan)) * c:.2f})")
    print(f"   residual    sigma_log {resid_sigma:.3f}  "
          f"=> typical error {np.exp(resid_sigma):.2f}x")

    # -- 3. is the remainder predictable? ---------------------------------
    print("\n3. Residual structure (model-free variance explained)")
    by_n = _explained(resid, n_unreached)
    by_mode = _explained(resid, mode)
    print(f"   by n_unreached   {by_n:6.1%}   "
          f"(a count the heuristic already scales with)")
    print(f"   by chosen mode   {by_mode:6.1%}   (coarse vs precise)")

    print("\n   ratio by robots still unreached:")
    print(f"   {'n':>4} {'count':>7} {'median':>9} {'sigma_log':>10}")
    for lo, hi in ((1, 3), (4, 6), (7, 9), (10, 12), (13, 14)):
        m = (n_unreached >= lo) & (n_unreached <= hi)
        if m.sum() < 10:
            continue
        mm, ss = _log_spread(log_ratio[m])
        print(f"   {f'{lo}-{hi}':>4} {int(m.sum()):>7} {np.exp(mm):>9.4f} "
              f"{ss:>10.3f}")

    # -- verdict ----------------------------------------------------------
    print("\n" + "=" * 72)
    if resid_sigma < _CONSTANT_IS_ENOUGH:
        print("VERDICT: a constant is enough. Retune alpha and skip stage 2.")
        print(f"  Residual sigma_log {resid_sigma:.3f} < {_CONSTANT_IS_ENOUGH}, "
              f"so a learned leaf has little headroom to win.")
        print(f"  Set the heuristic's alpha to {float(d.get('alpha', np.nan)) * c:.2f} "
              f"(or scale cost_to_go by {c:.4f}) and re-evaluate.")
    else:
        print("VERDICT: build the value head (stage 2).")
        print(f"  A constant leaves sigma_log {resid_sigma:.3f} "
              f"({np.exp(resid_sigma):.2f}x typical error), so there is real "
              f"state-dependent structure to learn.")
        print(f"  Fit the correction in LOG space against target "
              f"log(G / h_analytic); a constant baseline is log({c:.4f}).")
        if by_n > 0.25:
            print(f"  Note: {by_n:.0%} of the residual is explained by "
                  f"n_unreached alone — make sure the features carry it.")
    print("=" * 72)
    print("\nCaveat: labels are the realised cost of the policy that collected "
          "them,\nso re-measure after the policy changes materially.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, nargs="+", required=True,
                    help="shard files or directories from collect_leaf_data.py")
    args = ap.parse_args()
    report(load_shards(args.data))


if __name__ == "__main__":
    main()
