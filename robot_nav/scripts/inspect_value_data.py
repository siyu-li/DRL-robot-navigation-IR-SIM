"""
Inspect value_data_*.npz shards — label distribution and meta.

Usage:
    python -m robot_nav.scripts.inspect_value_data --data data/value_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/value_data")
    args = ap.parse_args()

    paths = sorted(Path(args.data).glob("value_data_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no value_data_*.npz shards in {args.data}")

    parts = [np.load(p) for p in paths]
    label_raw = np.concatenate([p["label"] for p in parts])   # cost units
    episode    = np.concatenate([p["episode"] for p in parts])
    decision   = np.concatenate([p["decision"] for p in parts])

    meta = {k[len("meta_"):]: float(parts[0][k]) for k in parts[0].files if k.startswith("meta_")}
    precise_cost = meta["precise_cost"]
    y = label_raw / precise_cost    # decision units (same as training)

    print(f"Shards : {len(paths)}")
    print(f"Samples: {len(y)}")
    print(f"Episodes: {np.unique(episode).size}")
    print(f"\nMeta: {meta}")

    print("\n--- Label distribution (decision units) ---")
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pct_vals = np.percentile(y, percentiles)
    for p, v in zip(percentiles, pct_vals):
        print(f"  p{p:3d}: {v:.2f}")
    print(f"  mean : {y.mean():.3f}")
    print(f"  std  : {y.std():.3f}")
    print(f"  MAE/mean ratio (straight-line ~1.66 / mean): {1.662 / y.mean():.3f}  "
          f"(embed ~1.71 / mean): {1.711 / y.mean():.3f}")

    print("\n--- Decision index distribution (how early/late in episode samples come from) ---")
    for p in [0, 25, 50, 75, 100]:
        print(f"  p{p:3d}: decision {np.percentile(decision, p):.0f}")

    # ASCII histogram of labels
    print("\n--- Label histogram (decision units) ---")
    counts, edges = np.histogram(y, bins=20)
    bar_max = 40
    scale = bar_max / counts.max()
    for i, c in enumerate(counts):
        bar = "#" * int(c * scale)
        print(f"  [{edges[i]:5.1f}-{edges[i+1]:5.1f}]: {bar} ({c})")


if __name__ == "__main__":
    main()
