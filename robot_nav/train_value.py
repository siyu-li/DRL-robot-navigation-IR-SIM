"""
Train the per-robot precise cost-to-go regressor v_psi (CAPSwitcher value step 1).

Supervised regression on the data written by ``robot_nav/collect_value_data.py``:
per-robot feature -> remaining precise decisions (labels are stored in cost
units, ``decisions * precise_cost``; training happens in decision units for
conditioning and the checkpoint carries ``precise_cost`` to scale back).

Two feature variants are trained and compared on a held-out EPISODE split
(samples of one episode never straddle train/val):

  * ``embedding`` — frozen GAT per-robot 512-d embedding (neighbor-aware);
  * ``geometry``  — raw 11-col state row (congestion-blind baseline).

For context the report includes the crude analytic heuristic's per-robot
prediction on the same validation states, expressed in decision units:
``dist / (lin_max * step_time)`` — what the current cost_to_go charges per
robot — and the idealized ``dist / (lin_max * step_time * selection_interval)``
(straight-line decisions).

Sim-free: runs locally on saved shards.
    python -m robot_nav.train_value --data data/value_data --out-dir robot_nav/models/MARL/capswitcher/checkpoint/value
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robot_nav.models.MARL.capswitcher.rl.value_net import (
    PerRobotValue,
    save_value_checkpoint,
)

_HIDDEN = {"embedding": (256, 128), "geometry": (64, 64)}


def load_shards(data_dir: Path) -> dict:
    """Concatenate all value_data_*.npz shards; meta is read from the first."""
    paths = sorted(data_dir.glob("value_data_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no value_data_*.npz shards in {data_dir}")
    parts = [np.load(p) for p in paths]
    data = {
        k: np.concatenate([p[k] for p in parts])
        for k in ("emb", "geo", "label", "robot", "decision", "episode")
    }
    data["meta"] = {
        k[len("meta_"):]: float(parts[0][k]) for k in parts[0].files if k.startswith("meta_")
    }
    print(f"Loaded {data['label'].shape[0]} samples from {len(paths)} shard(s); meta={data['meta']}")
    return data


def episode_split(episode: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean validation mask, split at the episode level."""
    uniq = np.unique(episode)
    rng = np.random.default_rng(seed)
    val_eps = rng.permutation(uniq)[: max(1, int(round(val_frac * uniq.size)))]
    return np.isin(episode, val_eps)


def train_one(
    feature: str,
    x: np.ndarray,
    y: np.ndarray,
    val_mask: np.ndarray,
    precise_cost: float,
    meta: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PerRobotValue, dict]:
    """Train one regressor variant; returns (net, report) with val metrics."""
    x_tr, y_tr = x[~val_mask], y[~val_mask]
    x_va, y_va = x[val_mask], y[val_mask]

    x_mean = x_tr.mean(axis=0)
    x_std = np.maximum(x_tr.std(axis=0), 1e-6)
    xt = torch.as_tensor((x_tr - x_mean) / x_std, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    xv = torch.as_tensor((x_va - x_mean) / x_std, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_va, dtype=torch.float32, device=device)

    net = PerRobotValue(in_dim=x.shape[1], hidden=_HIDDEN[feature]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n = xt.shape[0]
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            net.eval()
            with torch.no_grad():
                pred = net(xv).clamp_min(0.0)
                mse = float(loss_fn(pred, yv))
                mae = float((pred - yv).abs().mean())
            print(f"  [{feature}] epoch {epoch:3d}: val MSE={mse:.4f}  MAE={mae:.3f} decisions")

    net.eval()
    with torch.no_grad():
        pred = net(xv).clamp_min(0.0)
        report = {
            "val_mae_decisions": float((pred - yv).abs().mean()),
            "val_rmse_decisions": float(((pred - yv) ** 2).mean().sqrt()),
            "val_samples": int(yv.shape[0]),
        }
    save_value_checkpoint(
        Path(args.out_dir) / f"value_{feature}.pt",
        net, feature, x.shape[1], _HIDDEN[feature], x_mean, x_std, precise_cost,
        meta={**meta, **report, "epochs": args.epochs},
    )
    return net, report


def crude_baselines(geo_val: np.ndarray, y_val: np.ndarray, meta: dict) -> dict:
    """MAE (decision units) of the analytic heuristics on the validation states."""
    dist = np.linalg.norm(geo_val[:, [9, 10]] - geo_val[:, [0, 1]], axis=1)
    per_substep = meta["lin_max"] * meta["step_time"]
    crude = dist / per_substep                                     # current h-hat per robot
    straight = dist / (per_substep * meta["selection_interval"])   # idealized decisions
    return {
        "crude_hhat_mae": float(np.abs(crude - y_val).mean()),
        "straightline_mae": float(np.abs(straight - y_val).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/value_data")
    ap.add_argument("--out-dir", type=str,
                    default="robot_nav/models/MARL/capswitcher/checkpoint/value")
    ap.add_argument("--feature", choices=["embedding", "geometry", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-label", type=int, default=None,
                    help="discard samples with label > this many decisions "
                         "(local embedding can only see a limited view; "
                         "long-horizon labels are not predictable from local features)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_shards(Path(args.data))
    meta = data["meta"]
    precise_cost = meta["precise_cost"]
    y = data["label"] / precise_cost                     # train in decision units

    # episode-level split on the FULL data first, then apply label filter —
    # this preserves the episode-level non-overlap guarantee.
    val_mask = episode_split(data["episode"], args.val_frac, args.split_seed)

    if args.max_label is not None:
        keep = y <= args.max_label
        dropped = int((~keep).sum())
        y        = y[keep]
        val_mask = val_mask[keep]
        for k in ("emb", "geo"):
            data[k] = data[k][keep]
        print(f"Label filter: keeping y <= {args.max_label} decisions "
              f"({int(keep.sum())} kept, {dropped} dropped)")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"Train/val: {int((~val_mask).sum())}/{int(val_mask.sum())} samples "
        f"(episode-level split), label mean={y.mean():.2f} max={y.max():.0f} decisions"
    )

    features = ["embedding", "geometry"] if args.feature == "both" else [args.feature]
    reports: dict[str, dict] = {}
    for feature in features:
        x = data["emb"] if feature == "embedding" else data["geo"]
        print(f"\nTraining {feature} regressor (in_dim={x.shape[1]}) ...")
        _, reports[feature] = train_one(
            feature, x, y, val_mask, precise_cost, meta, args, device
        )

    base = crude_baselines(data["geo"][val_mask], y[val_mask], meta)
    print("\n=== Validation MAE (decision units) ===")
    for feature, rep in reports.items():
        print(f"  learned {feature:<9}: {rep['val_mae_decisions']:.3f}")
    print(f"  crude h-hat (alpha*dist, per robot): {base['crude_hhat_mae']:.3f}")
    print(f"  straight-line decisions:             {base['straightline_mae']:.3f}")
    print(f"\nCheckpoints in {args.out_dir}/ — use with "
          f"eval_mpc --value-model <path to value_<feature>.pt>")


if __name__ == "__main__":
    main()
