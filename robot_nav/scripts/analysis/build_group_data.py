"""
Build group-level (2-robot pair) dataset from per-robot collected data.

For every snapshot t and every unique pair (i, j) with i < j,
this script constructs one group record with:

  Pooled embeddings (3 strategies × 4 embedding types):
    {emb}_mean  — (emb_i + emb_j) / 2
    {emb}_max   — elementwise max(emb_i, emb_j)
    {emb}_diff  — |emb_i - emb_j|   (captures within-pair contrast)

  Per-pair scalars:
    dtg_a / dtg_b          — individual dist-to-goal
    mean_dtg               — (dtg_a + dtg_b) / 2
    density_s2_a/b         — Gaussian density score σ=2m per robot
    density_s4_a/b         — Gaussian density score σ=4m per robot
    mean_density_s2/s4     — mean of the pair
    intra_dist             — centre-to-centre dist between the two robots (m)
    intra_prox             — boundary-to-boundary proximity (m), = intra_dist - 2*R

  Category labels (computed from global p33/p67 thresholds):
    goal_category (int8)
        0 — both close     (both dtg < p33)
        1 — both far       (both dtg > p67)
        2 — both middle    (both dtg in [p33, p67])
        3 — mixed          (one < p33, other > p67)
       -1 — unassigned

    density_cat_s2 / density_cat_s4 (int8)
        0 — both sparse    (both density < p33)
        1 — both dense     (both density > p67)
        2 — mixed          (one < p33, other > p67)
       -1 — unassigned

Input:   robot_nav/analysis/collected_data/gat_data.npz
Output:  robot_nav/analysis/collected_data/group_data.npz

Usage:
    python -m robot_nav.scripts.analysis.build_group_data
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np


# =====================================================================
# Configuration
# =====================================================================
CONFIG = {
    "data_path": "robot_nav/analysis/collected_data/gat_data.npz",
    "save_path": "robot_nav/analysis/collected_data/group_data.npz",
}

ROBOT_RADIUS = 0.2


# =====================================================================
# Category helpers
# =====================================================================
def assign_goal_category(
    dtg_a: np.ndarray,
    dtg_b: np.ndarray,
    p33: float,
    p67: float,
) -> np.ndarray:
    """
    4-way goal category for each pair.

    Returns int8 array:
      0 — both close     (both < p33)
      1 — both far       (both > p67)
      2 — both middle    (both in [p33, p67])
      3 — mixed          (one < p33, other > p67)
     -1 — unassigned     (all other combinations)
    """
    cats = np.full(len(dtg_a), -1, dtype=np.int8)
    close_a = dtg_a < p33;   close_b = dtg_b < p33
    far_a   = dtg_a > p67;   far_b   = dtg_b > p67
    mid_a   = ~close_a & ~far_a
    mid_b   = ~close_b & ~far_b
    cats[close_a & close_b] = 0
    cats[far_a   & far_b  ] = 1
    cats[mid_a   & mid_b  ] = 2
    cats[(close_a & far_b) | (far_a & close_b)] = 3
    return cats


def assign_density_category(
    dens_a: np.ndarray,
    dens_b: np.ndarray,
    p33: float,
    p67: float,
) -> np.ndarray:
    """
    3-way density category for each pair.

    Returns int8 array:
      0 — both sparse    (both < p33)
      1 — both dense     (both > p67)
      2 — mixed          (one < p33, other > p67)
     -1 — unassigned
    """
    cats = np.full(len(dens_a), -1, dtype=np.int8)
    sparse_a = dens_a < p33;  sparse_b = dens_b < p33
    dense_a  = dens_a > p67;  dense_b  = dens_b > p67
    cats[sparse_a & sparse_b] = 0
    cats[dense_a  & dense_b ] = 1
    cats[(sparse_a & dense_b) | (dense_a & sparse_b)] = 2
    return cats


# =====================================================================
# Main builder
# =====================================================================
def build_group_data(data_path: str, save_path: str) -> None:
    print("=" * 60)
    print("Building group (2-robot pair) dataset")
    print("=" * 60)

    # ---- Load per-robot data ----
    print(f"\n[1/4] Loading {data_path} ...")
    raw = np.load(data_path)

    S, N = raw["dist_to_goal"].shape
    print(f"  Snapshots: {S}  |  Robots: {N}")

    # ---- All unique pairs ----
    pairs = np.array(list(combinations(range(N), 2)))   # (P, 2)
    P = len(pairs)
    a_idx, b_idx = pairs[:, 0], pairs[:, 1]
    print(f"  Pairs/snapshot: C({N},2) = {P}  →  total rows: {S * P:,}")

    # ---- Global percentile thresholds ----
    print("\n[2/4] Computing global thresholds ...")
    dtg_all    = raw["dist_to_goal"].ravel()
    dens2_all  = raw["density_sigma2"].ravel()
    dens4_all  = raw["density_sigma4"].ravel()

    dtg_p33,   dtg_p67   = np.percentile(dtg_all,  [33, 67])
    dens2_p33, dens2_p67 = np.percentile(dens2_all,[33, 67])
    dens4_p33, dens4_p67 = np.percentile(dens4_all,[33, 67])

    print(f"  dist_to_goal  p33={dtg_p33:.3f}  p67={dtg_p67:.3f}")
    print(f"  density σ=2m  p33={dens2_p33:.3f}  p67={dens2_p67:.3f}")
    print(f"  density σ=4m  p33={dens4_p33:.3f}  p67={dens4_p67:.3f}")

    # ---- Vectorised pair-level scalar arrays ----
    print("\n[3/4] Building pair-level arrays ...")

    # Shape: (S, P) → flatten to (S*P,)
    dtg_a   = raw["dist_to_goal"][:,  a_idx].ravel()
    dtg_b   = raw["dist_to_goal"][:,  b_idx].ravel()
    dens2_a = raw["density_sigma2"][:, a_idx].ravel()
    dens2_b = raw["density_sigma2"][:, b_idx].ravel()
    dens4_a = raw["density_sigma4"][:, a_idx].ravel()
    dens4_b = raw["density_sigma4"][:, b_idx].ravel()

    # Intra-pair distance: dist_rr is (S, N, N)
    intra_dist = raw["dist_rr"][:, a_idx, b_idx].ravel()   # (S*P,)
    intra_prox = intra_dist - 2 * ROBOT_RADIUS

    snap_ids = np.repeat(np.arange(S), P)   # (S*P,)
    robot_a  = np.tile(a_idx, S)            # (S*P,)
    robot_b  = np.tile(b_idx, S)            # (S*P,)

    goal_cat     = assign_goal_category(dtg_a, dtg_b, dtg_p33, dtg_p67)
    density_cat2 = assign_density_category(dens2_a, dens2_b, dens2_p33, dens2_p67)
    density_cat4 = assign_density_category(dens4_a, dens4_b, dens4_p33, dens4_p67)

    # ---- Pooled embeddings ----
    emb_keys = {
        "self_embedding": 256,
        "attn_embedding": 256,
        "pre_decode":     512,
        "full_embedding": 512,
    }
    pooled: dict = {}
    for key in emb_keys:
        E  = raw[key]               # (S, N, D)
        ea = E[:, a_idx, :]         # (S, P, D)
        eb = E[:, b_idx, :]         # (S, P, D)
        D  = E.shape[2]
        pooled[f"{key}_mean"] = ((ea + eb) / 2).reshape(S * P, D)
        pooled[f"{key}_max"]  = np.maximum(ea, eb).reshape(S * P, D)
        pooled[f"{key}_diff"] = np.abs(ea - eb).reshape(S * P, D)
        print(f"  {key}: mean/max/diff  shape ({S*P}, {D})")

    # ---- Save ----
    print(f"\n[4/4] Saving to {save_path} ...")
    save_dict = {
        # Index
        "snapshot_id":      snap_ids,
        "robot_a":          robot_a,
        "robot_b":          robot_b,
        # Per-robot scalars
        "dtg_a":            dtg_a,
        "dtg_b":            dtg_b,
        "mean_dtg":         (dtg_a + dtg_b) / 2,
        "density_s2_a":     dens2_a,
        "density_s2_b":     dens2_b,
        "density_s4_a":     dens4_a,
        "density_s4_b":     dens4_b,
        "mean_density_s2":  (dens2_a + dens2_b) / 2,
        "mean_density_s4":  (dens4_a + dens4_b) / 2,
        "intra_dist":       intra_dist,
        "intra_prox":       intra_prox,
        # Category labels
        "goal_category":    goal_cat,
        "density_cat_s2":   density_cat2,
        "density_cat_s4":   density_cat4,
        # Stored thresholds (metadata for downstream scripts)
        "dtg_p33":          np.array(dtg_p33),
        "dtg_p67":          np.array(dtg_p67),
        "dens2_p33":        np.array(dens2_p33),
        "dens2_p67":        np.array(dens2_p67),
        "dens4_p33":        np.array(dens4_p33),
        "dens4_p67":        np.array(dens4_p67),
        **pooled,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **save_dict)
    size_mb = Path(save_path).stat().st_size / 1024 / 1024
    print(f"  Saved  ({size_mb:.1f} MB)")

    # ---- Category distribution summary ----
    print("\nGoal category distribution:")
    total = len(goal_cat)
    for lbl, name in [(0, "both-close"), (1, "both-far"),
                      (2, "both-middle"), (3, "mixed"), (-1, "unassigned")]:
        cnt = int((goal_cat == lbl).sum())
        print(f"  [{lbl:2d}] {name:<15}  {cnt:>8,}  ({100*cnt/total:.1f}%)")

    print("\nDensity category (σ=2m) distribution:")
    for lbl, name in [(0, "both-sparse"), (1, "both-dense"),
                      (2, "mixed"), (-1, "unassigned")]:
        cnt = int((density_cat2 == lbl).sum())
        print(f"  [{lbl:2d}] {name:<15}  {cnt:>8,}  ({100*cnt/total:.1f}%)")
    print("=" * 60)


# =====================================================================
# Entry point
# =====================================================================
def main():
    cfg = CONFIG
    build_group_data(cfg["data_path"], cfg["save_path"])


if __name__ == "__main__":
    main()
