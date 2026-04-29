"""
MLP classification probes for group (2-robot pair) embeddings.

This script evaluates whether a small MLP (2–3 hidden layers) can classify:
    1) 4-way goal category (`goal_category`)
    2) 3-way density category (`density_cat_s2`, same as visualize_group_tsne)

For each embedding × pooling combination, it trains a balanced classifier
and reports:
    - Overall accuracy
    - Per-class precision / recall / F1
    - Confusion matrix (printed and saved as heatmap)

Uses the same group_data.npz produced by build_group_data.py.

Usage:
        python -m robot_nav.scripts.analysis.classify_group_mlp
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


# =====================================================================
# Configuration
# =====================================================================
CONFIG = {
    "data_path": "robot_nav/analysis/collected_data/group_data.npz",
    "save_dir":  "robot_nav/analysis/group_mlp_probe",

    # Which (embedding, pooling) to probe
    "runs": [
        ("full_embedding", "mean"),
        ("full_embedding", "diff"),
        ("full_embedding", "max"),
        ("pre_decode",     "mean"),
        ("pre_decode",     "diff"),
        ("self_embedding", "mean"),
        ("attn_embedding", "mean"),
    ],

    # MLP architecture
    "hidden_dims": [256, 128],
    "dropout": 0.1,

    # Training
    "epochs": 60,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "train_ratio": 0.8,

    # Balanced sampling: max samples per class
    "max_per_class": 3000,

    "random_seed": 42,
    "dpi": 150,
}

# Label names
GOAL_LABELS = {0: "both-close", 1: "both-far", 2: "both-mid", 3: "mixed"}
DENSITY_LABELS = {0: "both-sparse", 1: "both-dense", 2: "mixed"}


# =====================================================================
# MLP model
# =====================================================================
class ProbeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int],
                 n_classes: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =====================================================================
# Balanced sampling
# =====================================================================
def balanced_sample(labels: np.ndarray, max_per_class: int,
                    rng: np.random.Generator) -> np.ndarray:
    valid = sorted(int(c) for c in np.unique(labels) if c >= 0)
    n = min(min(int((labels == c).sum()) for c in valid), max_per_class)
    parts = []
    for c in valid:
        idx = np.where(labels == c)[0]
        parts.append(rng.choice(idx, size=n, replace=False))
    return np.concatenate(parts)


# =====================================================================
# Train + evaluate one run
# =====================================================================
def run_probe(
    emb_key: str,
    pool: str,
    data,
    cfg: dict,
    save_dir: Path,
    device: torch.device,
    rng: np.random.Generator,
    target_key: str,
    label_map: dict,
    task_tag: str,
) -> dict:
    arr_key  = f"{emb_key}_{pool}"
    X_all    = data[arr_key]
    y_all    = data[target_key]

    # ---- Balanced sample ----
    idx      = balanced_sample(y_all, cfg["max_per_class"], rng)
    rng.shuffle(idx)
    X, y     = X_all[idx], y_all[idx]

    # Remap labels to contiguous 0..3  (they already are 0,1,2,3 but filter -1)
    mask     = y >= 0
    X, y     = X[mask], y[mask]
    n_classes = int(y.max()) + 1

    # ---- Train / test split ----
    n_train  = int(len(X) * cfg["train_ratio"])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # ---- Standardise (fit on train) ----
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_test  = (X_test  - mu) / std

    # ---- To tensors ----
    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_test,  dtype=torch.long,    device=device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    # ---- Model ----
    input_dim = X_train.shape[1]
    model = ProbeClassifier(input_dim, cfg["hidden_dims"], n_classes,
                            cfg["dropout"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    model.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        for xb, yb in train_dl:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 20 == 0:
            avg = total_loss / len(X_tr_t)
            print(f"      epoch {epoch+1:3d}/{cfg['epochs']}  loss={avg:.4f}")

    # ---- Evaluation ----
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        preds  = logits.argmax(dim=1).cpu().numpy()
    y_true = y_te_t.cpu().numpy()

    acc = float((preds == y_true).mean())
    label_names = [label_map[i] for i in range(n_classes)]

    report = classification_report(y_true, preds, target_names=label_names,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=list(range(n_classes)))

    # ---- Print ----
    tag = f"{emb_key} × {pool}"
    full_tag = f"{task_tag} | {tag}"
    print(f"\n  [{tag}]  test acc = {acc:.3f}  "
          f"(train={len(X_train)}, test={len(X_test)})")
    print(report)

    # ---- Confusion matrix heatmap ----
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(label_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion — {full_tag}\nacc={acc:.3f}", fontsize=10)
    fig.tight_layout()
    fname = f"cm_{task_tag}_{emb_key}_{pool}.png"
    fig.savefig(save_dir / fname, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}")

    return {"tag": tag, "accuracy": acc, "cm": cm, "report": report, "task": task_tag}


def run_task(
    *,
    task_title: str,
    task_tag: str,
    target_key: str,
    label_map: dict,
    cfg: dict,
    data,
    save_dir: Path,
    device: torch.device,
    rng: np.random.Generator,
) -> list[dict]:
    print("=" * 60)
    print(task_title)
    print("=" * 60)

    y = data[target_key]
    total = len(y)
    print(f"\nTotal rows: {total:,}")
    for lbl, name in label_map.items():
        cnt = int((y == lbl).sum())
        print(f"  [{lbl}] {name:<15}  {cnt:>8,}  ({100*cnt/total:.1f}%)")
    print(f"  [-1] unassigned     {int((y == -1).sum()):>8,}")

    results = []
    for emb_key, pool in cfg["runs"]:
        print(f"\n{'─'*60}")
        print(f"  Training: {emb_key} × {pool}")
        print(f"    Input dim: {data[f'{emb_key}_{pool}'].shape[1]}")
        r = run_probe(
            emb_key,
            pool,
            data,
            cfg,
            save_dir,
            device,
            rng,
            target_key=target_key,
            label_map=label_map,
            task_tag=task_tag,
        )
        results.append(r)

    print(f"\n{'='*60}")
    print(f"SUMMARY — {task_title}")
    print(f"{'='*60}")
    print(f"  {'Configuration':<35}  {'Accuracy':>8}")
    print(f"  {'─'*35}  {'─'*8}")
    for r in results:
        print(f"  {r['tag']:<35}  {r['accuracy']:>8.3f}")
    print(f"{'='*60}")
    return results


# =====================================================================
# Main
# =====================================================================
def main():
    cfg      = CONFIG
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    rng      = np.random.default_rng(cfg["random_seed"])
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(cfg["data_path"])
    run_task(
        task_title="MLP Classification Probe — Goal Category (4-way)",
        task_tag="goal4",
        target_key="goal_category",
        label_map=GOAL_LABELS,
        cfg=cfg,
        data=data,
        save_dir=save_dir,
        device=device,
        rng=rng,
    )

    run_task(
        task_title="MLP Classification Probe — Density Category (3-way, sigma=2m)",
        task_tag="density3",
        target_key="density_cat_s2",
        label_map=DENSITY_LABELS,
        cfg=cfg,
        data=data,
        save_dir=save_dir,
        device=device,
        rng=rng,
    )

    print(f"\nAll confusion matrices saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
