"""
Cost on the COMMON successful episodes across policies.

``cost/success ep`` in the eval tables averages over each policy's *own*
successes, so two policies' numbers describe different episode sets — a
policy that only solves the easy episodes looks cheap.  This tool joins the
per-episode records saved in eval shards on the episode seed and averages
only over the episodes **every** compared policy solved, so the cost
comparison is on identical navigation problems.

Reads the ``per_episode`` lists saved by ``eval_gaz14_lazy --out`` and
``eval_gaz14_baselines --out`` shards.  Shards written before per-episode
records existed are skipped with a warning — re-run those to include them.

Usage:
    python -m robot_nav.compare_common_eps runs/gaz14_lazy_sweep runs/baselines14
    python -m robot_nav.compare_common_eps runs/gaz14_lazy_sweep --algos GAZ14-b100+p GAZ14-b400+p
    python -m robot_nav.compare_common_eps runs/gaz14_lazy_sweep --list-seeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_nav.eval_gaz14_lazy import print_table

COMMON_ROWS = [
    ("episodes covered",       "episodes",            "{:d}"),
    ("success rate",           "success_rate",        "{:.1%}"),
    ("cost/common ep  <<",     "avg_cost_common",     "{:.0f}"),
    ("  coarse cost",          "avg_coarse_common",   "{:.0f}"),
    ("  precise cost",         "avg_precise_common",  "{:.0f}"),
    ("  precise share",        "precise_share_common", "{:.1%}"),
    ("decisions/common ep",    "avg_dec_common",      "{:.1f}"),
    ("  precise fraction",     "precise_frac_common", "{:.1%}"),
    ("transitions/common ep",  "avg_trans_common",    "{:.0f}"),
]


def load_records(dirs: list[str]) -> dict[str, dict[int, dict]]:
    """``label -> {seed -> episode record}`` from every shard in ``dirs``."""
    by_algo: dict[str, dict[int, dict]] = {}
    skipped: list[str] = []
    for d in dirs:
        paths = sorted(Path(d).glob("*.json"))
        if not paths:
            print(f"WARNING: no .json shards in {d}")
        for p in paths:
            shard = json.loads(p.read_text())
            if not isinstance(shard, dict) or "label" not in shard:
                continue                      # e.g. a results.json, not a shard
            if not shard.get("per_episode"):
                skipped.append(str(p))
                continue
            recs = by_algo.setdefault(shard["label"], {})
            for rec in shard["per_episode"]:
                if rec["seed"] in recs:
                    print(f"WARNING: duplicate seed {rec['seed']} for "
                          f"{shard['label']} ({p.name}) — keeping the first.")
                    continue
                recs[rec["seed"]] = rec
    if skipped:
        print(f"Skipped {len(skipped)} shard(s) without per-episode records "
              "(re-run to include them):")
        for s in skipped:
            print(f"  {s}")
    return by_algo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+",
                    help="shard directories (eval_gaz14_lazy / "
                         "eval_gaz14_baselines --out)")
    ap.add_argument("--algos", nargs="+", default=None,
                    help="restrict the comparison to these shard labels "
                         "(default: every label found)")
    ap.add_argument("--list-seeds", action="store_true",
                    help="also print the common-success seeds and, per "
                         "policy, the covered seeds it failed")
    args = ap.parse_args()

    by_algo = load_records(args.dirs)
    if args.algos:
        missing = [a for a in args.algos if a not in by_algo]
        if missing:
            raise SystemExit(
                f"labels not found: {missing}; available: {sorted(by_algo)}"
            )
        by_algo = {a: by_algo[a] for a in args.algos}
    if len(by_algo) < 2:
        raise SystemExit(
            f"need at least 2 policies to compare, found {sorted(by_algo)}"
        )

    # Join on the seeds every policy covered; compare on those it also solved.
    covered = set.intersection(*(set(r) for r in by_algo.values()))
    if not covered:
        raise SystemExit("no common covered seeds across the given policies")
    common = sorted(
        s for s in covered
        if all(r[s]["outcome"] == "SUCCESS" for r in by_algo.values())
    )
    print(f"\nPolicies: {list(by_algo)}")
    print(f"Common covered episodes: {len(covered)}"
          + (f" (union {len(set.union(*(set(r) for r in by_algo.values())))})"
             if len(set.union(*(set(r) for r in by_algo.values()))) != len(covered)
             else ""))
    print(f"Common SUCCESS episodes: {len(common)} "
          f"({len(common) / len(covered):.1%} of covered)")
    if not common:
        raise SystemExit("no episode was solved by every policy — nothing to average")

    results: dict[str, dict] = {}
    for label, recs in by_algo.items():
        rows = [recs[s] for s in common]
        n = len(rows)
        dec = sum(r["decisions"] for r in rows)
        cost = sum(r["cost"] for r in rows)
        precise_cost = sum(r["precise_cost"] for r in rows)
        results[label] = {
            "episodes": len(recs),
            "success_rate": (
                sum(recs[s]["outcome"] == "SUCCESS" for s in covered)
                / len(covered)
            ),
            "avg_cost_common": cost / n,
            "avg_coarse_common": sum(r["coarse_cost"] for r in rows) / n,
            "avg_precise_common": precise_cost / n,
            "precise_share_common": precise_cost / max(cost, 1e-9),
            "avg_dec_common": dec / n,
            "precise_frac_common": (
                sum(r["precise_dec"] for r in rows) / max(dec, 1)
            ),
            "avg_trans_common": sum(r["transitions"] for r in rows) / n,
        }
    print_table(results, rows=COMMON_ROWS)

    if args.list_seeds:
        print(f"common-success seeds: {common}")
        for label, recs in by_algo.items():
            failed = sorted(
                s for s in covered if recs[s]["outcome"] != "SUCCESS"
            )
            print(f"{label} failed ({len(failed)}): "
                  + ", ".join(f"{s}[{recs[s]['outcome'][:4]}]" for s in failed))


if __name__ == "__main__":
    main()
