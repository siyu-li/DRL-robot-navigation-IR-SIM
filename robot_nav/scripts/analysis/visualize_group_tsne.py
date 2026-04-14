"""
t-SNE Visualization of Group (2-robot pair) Embeddings.

For each selected embedding × pooling combination, produces one figure
with 5 side-by-side scatter plots:

  1. Goal category       (4-way categorical)
  2. Mean dist-to-goal   (continuous)
  3. Density category    (3-way categorical, σ=2m by default)
  4. Mean density score  (continuous)
  5. Intra-pair proximity (continuous, boundary-to-boundary)

Data is balanced before t-SNE: equal number of samples from each
goal category (both-close / both-far / both-middle / mixed).
Unassigned pairs (-1) are excluded from categorical coloring but
retained in continuous coloring.

Input:   robot_nav/analysis/collected_data/group_data.npz
Output:  robot_nav/analysis/group_tsne/*.png

Usage:
    python -m robot_nav.scripts.analysis.visualize_group_tsne
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


# =====================================================================
# Configuration
# =====================================================================
CONFIG = {
    "data_path":  "robot_nav/analysis/collected_data/group_data.npz",
    "save_dir":   "robot_nav/analysis/group_tsne",

    # t-SNE
    "tsne_perplexity": 30,
    "random_state":    42,

    # Max samples per goal-category class (for balanced subsampling)
    # Total points fed to t-SNE = 4 × max_per_class
    "max_per_class": 800,

    # Density metric to use for categorical + continuous coloring
    "density_cat_key":  "density_cat_s2",     # 3-way labels
    "density_mean_key": "mean_density_s2",    # continuous scores

    # Which (embedding, pooling) pairs to visualize
    # Options: embedding in {self_embedding, attn_embedding, pre_decode, full_embedding}
    #          pooling  in {mean, max, diff}
    "runs": [
        ("full_embedding", "mean"),
        ("full_embedding", "diff"),
        ("pre_decode",     "mean"),
        ("pre_decode",     "diff"),
    ],

    "dpi": 150,
}

# =====================================================================
# Color palettes
# =====================================================================
GOAL_PALETTE = {
    0: ("#2196F3", "both-close"),
    1: ("#F44336", "both-far"),
    2: ("#FF9800", "both-middle"),
    3: ("#4CAF50", "mixed"),
}

DENSITY_PALETTE = {
    0: ("#9C27B0", "both-sparse"),
    1: ("#795548", "both-dense"),
    2: ("#00BCD4", "mixed"),
}


# =====================================================================
# Balanced subsampling
# =====================================================================
def balanced_sample(
    labels: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return indices of a class-balanced subsample (ignores label == -1).
    Draws min(class_size, max_per_class) samples from each valid class.
    """
    valid_classes = sorted(int(c) for c in np.unique(labels) if c >= 0)
    n = min(
        min(int((labels == c).sum()) for c in valid_classes),
        max_per_class,
    )
    idx_parts = []
    for c in valid_classes:
        c_idx = np.where(labels == c)[0]
        idx_parts.append(rng.choice(c_idx, size=n, replace=False))
    return np.concatenate(idx_parts)


# =====================================================================
# Scatter helpers
# =====================================================================
def _scatter_categorical(
    ax: plt.Axes,
    X2d: np.ndarray,
    labels: np.ndarray,
    palette: dict,
    title: str,
    s: float = 4,
    alpha: float = 0.65,
) -> None:
    for lbl, (color, name) in palette.items():
        mask = labels == lbl
        if mask.any():
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c=color, s=s, alpha=alpha, label=f"{name} (n={mask.sum()})")
    ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.7)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]);  ax.set_yticks([])


def _scatter_continuous(
    ax: plt.Axes,
    X2d: np.ndarray,
    values: np.ndarray,
    cmap: str,
    title: str,
    s: float = 4,
    alpha: float = 0.65,
) -> None:
    sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=values,
                    cmap=cmap, s=s, alpha=alpha)
    plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]);  ax.set_yticks([])


# =====================================================================
# Single t-SNE run
# =====================================================================
def run_and_plot(
    emb_key: str,
    pool: str,
    data,               # np.NpzFile
    cfg: dict,
    save_dir: Path,
    rng: np.random.Generator,
) -> None:
    arr_key   = f"{emb_key}_{pool}"
    X_all     = data[arr_key]                    # (S*P, D)
    goal_all  = data["goal_category"]            # (S*P,)  int8
    dens_all  = data[cfg["density_cat_key"]]     # (S*P,)  int8
    dtg_all   = data["mean_dtg"]                 # (S*P,)
    dens_c_all= data[cfg["density_mean_key"]]    # (S*P,)
    intra_all = data["intra_prox"]               # (S*P,)

    # ---- Balanced subsample (on goal category) ----
    sample_idx   = balanced_sample(goal_all, cfg["max_per_class"], rng)
    X_sub        = X_all[sample_idx]
    goal_sub     = goal_all[sample_idx]
    dens_sub     = dens_all[sample_idx]
    dtg_sub      = dtg_all[sample_idx]
    dens_cont_sub= dens_c_all[sample_idx]
    intra_sub    = intra_all[sample_idx]
    n_total      = len(X_sub)

    print(f"  [{emb_key:>15} × {pool:<4}]  {n_total} samples → t-SNE (perp={cfg['tsne_perplexity']}) ...")

    # ---- t-SNE ----
    X2d = TSNE(
        n_components=2,
        perplexity=cfg["tsne_perplexity"],
        random_state=cfg["random_state"],
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_sub)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    emb_label = emb_key.replace("_", " ")
    fig.suptitle(
        f"t-SNE  |  {emb_label}  ×  {pool}-pool  "
        f"({n_total} samples, balanced, perplexity={cfg['tsne_perplexity']})",
        fontsize=11, fontweight="bold",
    )

    _scatter_categorical(axes[0], X2d, goal_sub,  GOAL_PALETTE,
                         "Goal category (4-way)")
    _scatter_continuous (axes[1], X2d, dtg_sub,   "viridis",
                         "Mean dist-to-goal (m)")
    _scatter_categorical(axes[2], X2d, dens_sub,  DENSITY_PALETTE,
                         "Density category (3-way, σ=2m)")
    _scatter_continuous (axes[3], X2d, dens_cont_sub, "plasma",
                         "Mean density score (σ=2m)")
    _scatter_continuous (axes[4], X2d, intra_sub, "cividis",
                         "Intra-pair proximity (m)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f"tsne_{emb_key}_{pool}.png"
    fig.savefig(save_dir / fname, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}")


# =====================================================================
# Entry point
# =====================================================================
def main():
    cfg      = CONFIG
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    rng      = np.random.default_rng(cfg["random_state"])

    print("=" * 60)
    print("Group Embedding t-SNE Visualization")
    print("=" * 60)

    # ---- Load ----
    print(f"\nLoading {cfg['data_path']} ...")
    data  = np.load(cfg["data_path"])
    total = int(data["goal_category"].shape[0])

    # ---- Print category sizes ----
    gc = data["goal_category"]
    print(f"  Total pair-rows: {total:,}")
    print("\n  Goal category sizes (before balancing):")
    for lbl, name in [(0, "both-close"), (1, "both-far"),
                      (2, "both-middle"), (3, "mixed"), (-1, "unassigned")]:
        cnt = int((gc == lbl).sum())
        print(f"    [{lbl:2d}] {name:<15}  {cnt:>8,}  ({100*cnt/total:.1f}%)")

    dc = data[cfg["density_cat_key"]]
    print("\n  Density category sizes (σ=2m, before balancing):")
    for lbl, name in [(0, "both-sparse"), (1, "both-dense"),
                      (2, "mixed"), (-1, "unassigned")]:
        cnt = int((dc == lbl).sum())
        print(f"    [{lbl:2d}] {name:<15}  {cnt:>8,}  ({100*cnt/total:.1f}%)")

    # ---- t-SNE runs ----
    print(f"\nRunning {len(cfg['runs'])} t-SNE configuration(s) ...")
    for emb_key, pool in cfg["runs"]:
        run_and_plot(emb_key, pool, data, cfg, save_dir, rng)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {save_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
