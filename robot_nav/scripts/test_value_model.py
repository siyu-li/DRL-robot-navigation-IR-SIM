"""
Stratified error analysis for the learned value model(s).

Shows absolute MAE and relative MAE (RMAE = MAE/mean_true) per label bin,
compared against the straight-line analytic baseline — so you can see exactly
where the model beats or loses to the heuristic.

Usage:
    # test both checkpoints against the same val split used in training
    python -m robot_nav.scripts.test_value_model \
        --data  data/value_data \
        --ckpt  robot_nav/models/MARL/capswitcher/checkpoint/value/value_embedding.pt \
                robot_nav/models/MARL/capswitcher/checkpoint/value/value_geometry.pt

    # adjust val split seed/fraction to match the training run
    python -m robot_nav.scripts.test_value_model \
        --data data/value_data --val-frac 0.1 --split-seed 0 \
        --ckpt robot_nav/models/MARL/capswitcher/checkpoint/value/value_embedding.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from robot_nav.models.MARL.capswitcher.rl.value_net import load_value_checkpoint

# ── bins (decision units) ─────────────────────────────────────────────────────
BINS = [1, 3, 5, 8, 12, 20, 35, 60]   # right edges (last is open)
BIN_LABELS = [
    "1        ",
    "2 –  3   ",
    "4 –  5   ",
    "6 –  8   ",
    "9 – 12   ",
    "13 – 20  ",
    "21 – 35  ",
    "36+      ",
]


def load_shards(data_dir: Path) -> dict:
    paths = sorted(data_dir.glob("value_data_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no value_data_*.npz shards in {data_dir}")
    parts = [np.load(p) for p in paths]
    data = {k: np.concatenate([p[k] for p in parts])
            for k in ("emb", "geo", "label", "episode")}
    data["meta"] = {
        k[len("meta_"):]: float(parts[0][k])
        for k in parts[0].files if k.startswith("meta_")
    }
    return data


def episode_split(episode: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    uniq = np.unique(episode)
    rng = np.random.default_rng(seed)
    val_eps = rng.permutation(uniq)[: max(1, int(round(val_frac * uniq.size)))]
    return np.isin(episode, val_eps)


def label_bins(y: np.ndarray) -> np.ndarray:
    """Map each sample to a bin index 0..len(BINS)-1."""
    idx = np.zeros(len(y), dtype=int)
    for i, edge in enumerate(BINS[:-1]):
        idx[y > edge] = i + 1
    return idx


def straight_line_pred(geo: np.ndarray, meta: dict) -> np.ndarray:
    dist = np.linalg.norm(geo[:, [9, 10]] - geo[:, [0, 1]], axis=1)
    per_substep = meta["lin_max"] * meta["step_time"]
    return dist / (per_substep * meta["selection_interval"])


def predict(ckpt_path: Path, geo: np.ndarray, emb: np.ndarray,
            device: torch.device) -> tuple[str, np.ndarray]:
    net, ckpt = load_value_checkpoint(ckpt_path, device=device)
    x_mean = ckpt["x_mean"]
    x_std  = ckpt["x_std"]
    feature = ckpt["feature"]
    x_raw = emb if feature == "embedding" else geo
    x_norm = (x_raw - x_mean) / x_std
    xt = torch.as_tensor(x_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = net(xt).clamp_min(0.0).cpu().numpy()
    return feature, pred


def print_table(y: np.ndarray, preds: dict[str, np.ndarray], bin_idx: np.ndarray) -> None:
    col_w = 14
    names = list(preds.keys())
    header = f"  {'bin (decisions)':<16}" + "".join(f"{'n':>6}") + \
             "".join(f"{'mean_true':>{col_w}}") + \
             "".join(f"{n+' MAE':>{col_w}}" for n in names) + \
             "".join(f"{n+' RMAE':>{col_w}}" for n in names)
    print(header)
    print("  " + "-" * (16 + 6 + col_w * (1 + 2 * len(names))))

    overall_abs  = {n: [] for n in names}
    overall_true = []

    for b, label in enumerate(BIN_LABELS):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        yt = y[mask]
        n = mask.sum()
        mean_true = yt.mean()
        overall_true.extend(yt.tolist())

        mae_strs  = []
        rmae_strs = []
        for name in names:
            p = preds[name][mask]
            mae = float(np.abs(p - yt).mean())
            rmae = mae / mean_true if mean_true > 0 else float("nan")
            mae_strs.append(f"{mae:>{col_w}.3f}")
            rmae_strs.append(f"{rmae:>{col_w}.3f}")
            overall_abs[name].append(np.abs(p - yt))

        row = f"  {label:<16}{n:>6}{mean_true:>{col_w}.2f}"
        row += "".join(mae_strs) + "".join(rmae_strs)
        print(row)

    # overall
    print("  " + "-" * (16 + 6 + col_w * (1 + 2 * len(names))))
    overall_mean_true = float(np.mean(overall_true))
    mae_strs  = []
    rmae_strs = []
    for name in names:
        all_errs = np.concatenate(overall_abs[name])
        mae = float(all_errs.mean())
        rmae = mae / overall_mean_true
        mae_strs.append(f"{mae:>{col_w}.3f}")
        rmae_strs.append(f"{rmae:>{col_w}.3f}")
    row = f"  {'OVERALL':<16}{len(overall_true):>6}{overall_mean_true:>{col_w}.2f}"
    row += "".join(mae_strs) + "".join(rmae_strs)
    print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/value_data")
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one or more .pt checkpoint paths to compare")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="evaluate on ALL samples instead of val split only")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_shards(Path(args.data))
    meta = data["meta"]
    precise_cost = meta["precise_cost"]
    y = data["label"] / precise_cost     # decision units

    if args.all:
        mask = np.ones(len(y), dtype=bool)
        split_desc = "ALL samples"
    else:
        mask = episode_split(data["episode"], args.val_frac, args.split_seed)
        split_desc = f"val split ({mask.sum()} samples)"

    y_val   = y[mask]
    geo_val = data["geo"][mask]
    emb_val = data["emb"][mask]
    bin_idx = label_bins(y_val)

    preds: dict[str, np.ndarray] = {}

    # straight-line analytic baseline
    preds["straight"] = straight_line_pred(geo_val, meta)

    # learned checkpoints
    for ckpt_path in args.ckpt:
        feature, pred = predict(Path(ckpt_path), geo_val, emb_val, device)
        label = f"learned_{feature}"
        preds[label] = pred

    print(f"\nValue model stratified error — {split_desc}")
    print(f"precise_cost={precise_cost}, selection_interval={meta['selection_interval']}\n")
    print_table(y_val, preds, bin_idx)
    print()


if __name__ == "__main__":
    main()
