"""
t-SNE / UMAP Visualization of GAT Embeddings at Two Levels.

Level 1 — **Single-robot embeddings** (512-dim from frozen GAT):
    Scatter plots colored by per-robot targets:
        • dist_to_goal
        • heading_error
        • nearest_obstacle
        • nearest_robot
        • robot_id

Level 2 — **Group embeddings** (mean- or max-pooled over group members):
    Scatter plots colored by aggregated (mean / max) group targets:
        • dist_to_goal
        • heading_error
        • nearest_obstacle
        • nearest_robot

Usage:
    python -m robot_nav.scripts.visualize_gat_embeddings
"""

from __future__ import annotations

import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.groups.learned_action_coupling import (
    get_embeddings_from_frozen_actor,
)
from robot_nav.models.MARL.groups.group_generator import (
    generate_all_groups,
)
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

# Suppress IRSim warnings
from loguru import logger as loguru_logger
loguru_logger.disable("irsim")

# Try importing UMAP (optional dependency)
try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# =====================================================================
# Configuration — edit here
# =====================================================================
CONFIG = {
    # Decentralized model to load
    "model_name": "TD3-MARL-obstacle-14robots-partial-inactive_epoch210",
    "model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive",

    # Environment
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "num_robots": 14,
    "num_obstacles": 7,
    "state_dim": 11,
    "obstacle_state_dim": 4,

    # Data collection
    "n_snapshots": 500,        # Number of time-step snapshots to collect
    "max_steps_per_episode": 200,

    # Dimensionality reduction
    "method": "both",          # "tsne", "umap", or "both"
    "tsne_perplexity": 30,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "random_state": 42,

    # Group generation for level-2 visualization
    "group_sizes": [2, 3],     # Which group sizes to visualize
    "group_pooling": "both",   # "mean", "max", or "both"
    # If True, use structured groups from binary-allocation generator;
    # if False, enumerate all C(n,k) combinations (slow for size-3+).
    "use_structured_groups": True,

    # Output
    "save_dir": "robot_nav/analysis/gat_embedding_tsne_umap",
    "dpi": 150,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =====================================================================
# 1. Data Collection
# =====================================================================
def collect_embedding_data(
    policy: TD3Obstacle,
    sim: MARL_SIM_OBSTACLE,
    device: torch.device,
    n_snapshots: int = 500,
    max_steps: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Roll out the frozen policy and collect per-robot GAT embeddings
    paired with ground-truth labels.

    Returns dict with keys:
        embeddings      (S * N, 512)
        dist_to_goal    (S * N,)
        heading_error   (S * N,)
        nearest_obstacle(S * N,)
        nearest_robot   (S * N,)
        robot_id        (S * N,)
        snapshot_id     (S * N,)  — which snapshot each row belongs to
    """
    all_embeddings: List[np.ndarray] = []
    all_dist_to_goal: List[float] = []
    all_heading_error: List[float] = []
    all_nearest_obs: List[float] = []
    all_nearest_robot: List[float] = []
    all_robot_id: List[int] = []
    all_snapshot_id: List[int] = []

    (
        poses, distance, cos_val, sin_val, collision, goal, a, reward,
        positions, goal_positions, obstacle_states,
    ) = sim.reset(random_obstacles=True)

    collected = 0
    step = 0
    N = sim.num_robots

    while collected < n_snapshots:
        # Prepare observation
        robot_state, _ = policy.prepare_state(
            poses, distance, cos_val, sin_val, collision, a, goal_positions
        )
        robot_obs = np.array(robot_state)

        # Extract frozen embeddings — (N, 512)
        embeddings = get_embeddings_from_frozen_actor(
            policy.actor, robot_obs, obstacle_states, device
        ).cpu().numpy()

        # Get raw actions (for stepping the env)
        raw_action, combined_weights = policy.get_action(
            robot_obs, obstacle_states, add_noise=False
        )

        # Compute per-robot labels
        for i in range(N):
            px, py, theta = poses[i]
            gx, gy = goal_positions[i]

            # 1. Distance to goal
            dtg = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

            # 2. Heading error ∈ [0, π]
            desired_angle = np.arctan2(gy - py, gx - px)
            h_err = abs(desired_angle - theta)
            h_err = min(h_err, 2 * np.pi - h_err)

            # 3. Nearest obstacle distance
            obs_pos = obstacle_states[:, :2]  # (M, 2)
            obs_dists = np.sqrt(((obs_pos - np.array([px, py])) ** 2).sum(axis=1))
            nearest_obs = float(obs_dists.min()) if len(obs_dists) > 0 else 999.0

            # 4. Nearest robot distance
            robot_dists = []
            for j in range(N):
                if j != i:
                    robot_dists.append(
                        np.sqrt((poses[j][0] - px) ** 2 + (poses[j][1] - py) ** 2)
                    )
            nearest_robot = min(robot_dists) if robot_dists else 0.0

            all_embeddings.append(embeddings[i])
            all_dist_to_goal.append(dtg)
            all_heading_error.append(h_err)
            all_nearest_obs.append(nearest_obs)
            all_nearest_robot.append(nearest_robot)
            all_robot_id.append(i)
            all_snapshot_id.append(collected)

        collected += 1
        step += 1

        # Step environment forward
        a_in = [[(act[0] + 1) / 4, act[1]] for act in raw_action]
        (
            poses, distance, cos_val, sin_val, collision, goal, a, reward,
            positions, goal_positions, obstacle_states,
        ) = sim.step(a_in, None, combined_weights)

        if any(collision) or step >= max_steps:
            (
                poses, distance, cos_val, sin_val, collision, goal, a, reward,
                positions, goal_positions, obstacle_states,
            ) = sim.reset(random_obstacles=True)
            step = 0

    return {
        "embeddings": np.array(all_embeddings),
        "dist_to_goal": np.array(all_dist_to_goal),
        "heading_error": np.array(all_heading_error),
        "nearest_obstacle": np.array(all_nearest_obs),
        "nearest_robot": np.array(all_nearest_robot),
        "robot_id": np.array(all_robot_id),
        "snapshot_id": np.array(all_snapshot_id),
    }


# =====================================================================
# 2. Build Group Embeddings (Level 2)
# =====================================================================
def build_group_embeddings(
    data: Dict[str, np.ndarray],
    groups: List[List[int]],
    n_robots: int,
    pooling: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    For each snapshot, pool robot embeddings within each group.

    Args:
        data: Output of ``collect_embedding_data``.
        groups: List of groups (each a list of robot indices).
        n_robots: Number of robots.
        pooling: ``"mean"`` or ``"max"``.

    Returns dict with keys:
        embeddings         (S * G, 512)
        dist_to_goal       (S * G,)
        heading_error      (S * G,)
        nearest_obstacle   (S * G,)
        nearest_robot      (S * G,)
        group_size         (S * G,)   — number of robots in the group
        group_id           (S * G,)   — index of the group in `groups`
    """
    pool_fn = np.mean if pooling == "mean" else np.max

    emb = data["embeddings"]           # (S*N, D)
    n_total = emb.shape[0]
    n_snapshots = n_total // n_robots
    D = emb.shape[1]

    group_embeddings: List[np.ndarray] = []
    group_dtg: List[float] = []
    group_herr: List[float] = []
    group_nobs: List[float] = []
    group_nrob: List[float] = []
    group_sizes: List[int] = []
    group_ids: List[int] = []

    for t in range(n_snapshots):
        base = t * n_robots
        emb_t = emb[base: base + n_robots]           # (N, D)
        dtg_t = data["dist_to_goal"][base: base + n_robots]
        herr_t = data["heading_error"][base: base + n_robots]
        nobs_t = data["nearest_obstacle"][base: base + n_robots]
        nrob_t = data["nearest_robot"][base: base + n_robots]

        for gid, group in enumerate(groups):
            idx = list(group)
            # Pool embeddings
            g_emb = pool_fn(emb_t[idx], axis=0)       # (D,)
            # Pool targets
            g_dtg = pool_fn(dtg_t[idx])
            g_herr = pool_fn(herr_t[idx])
            g_nobs = pool_fn(nobs_t[idx])
            g_nrob = pool_fn(nrob_t[idx])

            group_embeddings.append(g_emb)
            group_dtg.append(g_dtg)
            group_herr.append(g_herr)
            group_nobs.append(g_nobs)
            group_nrob.append(g_nrob)
            group_sizes.append(len(group))
            group_ids.append(gid)

    return {
        "embeddings": np.array(group_embeddings),
        "dist_to_goal": np.array(group_dtg),
        "heading_error": np.array(group_herr),
        "nearest_obstacle": np.array(group_nobs),
        "nearest_robot": np.array(group_nrob),
        "group_size": np.array(group_sizes),
        "group_id": np.array(group_ids),
    }


# =====================================================================
# 3. Dimensionality Reduction
# =====================================================================
def reduce_dimensions(
    X: np.ndarray,
    method: str = "tsne",
    perplexity: float = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dim embeddings to 2-D.

    Args:
        X: (N, D) embedding matrix.
        method: ``"tsne"`` or ``"umap"``.

    Returns:
        (N, 2) reduced coordinates.
    """
    if method == "umap":
        if not HAS_UMAP:
            raise ImportError("umap-learn is not installed.  pip install umap-learn")
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )

    return reducer.fit_transform(X)


# =====================================================================
# 4. Plotting
# =====================================================================
def _scatter(
    ax: plt.Axes,
    X_2d: np.ndarray,
    colors: np.ndarray,
    label: str,
    cmap: str = "viridis",
    is_categorical: bool = False,
    s: float = 4,
    alpha: float = 0.6,
):
    """Draw a single scatter plot with colorbar or legend."""
    if is_categorical:
        unique_vals = np.unique(colors)
        cmap_obj = plt.cm.get_cmap("tab20", len(unique_vals))
        for k, val in enumerate(unique_vals):
            mask = colors == val
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[cmap_obj(k)], s=s, alpha=alpha,
                label=f"{label}={int(val)}",
            )
        ax.legend(
            fontsize=5, markerscale=2, ncol=2,
            loc="upper right", framealpha=0.7,
        )
    else:
        sc = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=colors, cmap=cmap, s=s, alpha=alpha,
        )
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)


def plot_level1(
    X_2d: np.ndarray,
    data: Dict[str, np.ndarray],
    method_name: str,
    save_dir: Path,
    dpi: int = 150,
):
    """
    Level 1 — single-robot embedding scatter plots.

    Produces one figure with 5 subplots (one per target).
    """
    targets = [
        ("dist_to_goal",      "Distance to Goal",   "viridis",  False),
        ("heading_error",      "Heading Error (rad)", "magma",   False),
        ("nearest_obstacle",   "Nearest Obstacle",   "cividis",  False),
        ("nearest_robot",      "Nearest Robot",      "plasma",   False),
        ("robot_id",           "Robot ID",           "tab20",    True),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    fig.suptitle(
        f"Level 1 — Single-Robot GAT Embeddings ({method_name})",
        fontsize=14, fontweight="bold",
    )

    for ax, (key, title, cmap, is_cat) in zip(axes, targets):
        _scatter(ax, X_2d, data[key], title, cmap=cmap, is_categorical=is_cat)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = save_dir / f"level1_robot_{method_name.lower()}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")

    # Also save each target as an individual high-res figure
    for key, title, cmap, is_cat in targets:
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        _scatter(
            ax_single, X_2d, data[key], title,
            cmap=cmap, is_categorical=is_cat, s=6, alpha=0.5,
        )
        ax_single.set_title(
            f"Robot Embedding — {title} ({method_name})", fontsize=12
        )
        ax_single.set_xlabel(f"{method_name} dim 1")
        ax_single.set_ylabel(f"{method_name} dim 2")
        out_single = save_dir / f"level1_{key}_{method_name.lower()}.png"
        fig_single.savefig(out_single, dpi=dpi, bbox_inches="tight")
        plt.close(fig_single)


def plot_level2(
    X_2d: np.ndarray,
    gdata: Dict[str, np.ndarray],
    pooling: str,
    method_name: str,
    save_dir: Path,
    dpi: int = 150,
):
    """
    Level 2 — group embedding scatter plots.

    Produces one figure with 5 subplots (4 targets + group_size).
    """
    agg_label = pooling.capitalize()  # "Mean" or "Max"
    targets = [
        ("dist_to_goal",      f"{agg_label} Dist to Goal",    "viridis",  False),
        ("heading_error",      f"{agg_label} Heading Error",   "magma",   False),
        ("nearest_obstacle",   f"{agg_label} Nearest Obstacle","cividis",  False),
        ("nearest_robot",      f"{agg_label} Nearest Robot",   "plasma",   False),
        ("group_size",         "Group Size",                   "Set1",     True),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    fig.suptitle(
        f"Level 2 — Group Embeddings ({pooling}-pool, {method_name})",
        fontsize=14, fontweight="bold",
    )

    for ax, (key, title, cmap, is_cat) in zip(axes, targets):
        _scatter(ax, X_2d, gdata[key], title, cmap=cmap, is_categorical=is_cat)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = save_dir / f"level2_group_{pooling}_{method_name.lower()}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")

    # Individual high-res figures
    for key, title, cmap, is_cat in targets:
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        _scatter(
            ax_single, X_2d, gdata[key], title,
            cmap=cmap, is_categorical=is_cat, s=6, alpha=0.5,
        )
        ax_single.set_title(
            f"Group Embedding — {title} ({method_name})", fontsize=12
        )
        ax_single.set_xlabel(f"{method_name} dim 1")
        ax_single.set_ylabel(f"{method_name} dim 2")
        out_single = save_dir / f"level2_{key}_{pooling}_{method_name.lower()}.png"
        fig_single.savefig(out_single, dpi=dpi, bbox_inches="tight")
        plt.close(fig_single)


# =====================================================================
# 5. Generate Groups
# =====================================================================
def generate_groups(
    num_robots: int,
    sizes: List[int],
    use_structured: bool = True,
) -> List[List[int]]:
    """
    Generate groups for level-2 visualization.

    Args:
        num_robots: Total number of robots.
        sizes: Which group sizes to include (e.g., [2, 3]).
        use_structured: If True use binary-allocation groups;
            otherwise use all C(n,k) combinations.
    """
    if use_structured:
        m = 4 if num_robots > 6 else 3
        all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)
        groups = [g for g in all_groups if len(g) in sizes]
    else:
        groups = []
        for k in sizes:
            for combo in combinations(range(num_robots), k):
                groups.append(list(combo))

    print(f"  Groups generated: {len(groups)} total")
    for s in sorted(set(sizes)):
        cnt = sum(1 for g in groups if len(g) == s)
        print(f"    size-{s}: {cnt}")
    return groups


# =====================================================================
# Main
# =====================================================================
def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GAT Embedding Visualization — t-SNE / UMAP")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load environment + policy
    # ------------------------------------------------------------------
    print("\n[1/5] Loading environment and frozen policy ...")
    sim = MARL_SIM_OBSTACLE(
        world_file=cfg["world_file"],
        disable_plotting=True,
    )

    policy = TD3Obstacle(
        state_dim=cfg["state_dim"],
        action_dim=2,
        max_action=1.0,
        device=device,
        num_robots=cfg["num_robots"],
        num_obstacles=cfg["num_obstacles"],
        obstacle_state_dim=cfg["obstacle_state_dim"],
        load_model=True,
        model_name=cfg["model_name"],
        load_model_name=cfg["model_name"],
        load_directory=Path(cfg["model_directory"]),
        save_directory=Path(cfg["model_directory"]),
        inference_only=True,
    )
    policy.actor.eval()
    print(f"  Model loaded: {cfg['model_name']}")
    print(f"  Robots: {sim.num_robots}  |  Obstacles: {sim.num_obstacles}")

    # ------------------------------------------------------------------
    # 2. Collect data
    # ------------------------------------------------------------------
    print(f"\n[2/5] Collecting {cfg['n_snapshots']} snapshots ...")
    data = collect_embedding_data(
        policy, sim, device,
        n_snapshots=cfg["n_snapshots"],
        max_steps=cfg["max_steps_per_episode"],
    )
    total_points = data["embeddings"].shape[0]
    embed_dim = data["embeddings"].shape[1]
    print(f"  Collected {total_points} robot embeddings  (dim={embed_dim})")

    # ------------------------------------------------------------------
    # 3. Determine which reduction methods to run
    # ------------------------------------------------------------------
    methods: List[str] = []
    if cfg["method"] in ("tsne", "both"):
        methods.append("tsne")
    if cfg["method"] in ("umap", "both"):
        if HAS_UMAP:
            methods.append("umap")
        else:
            print("  ⚠️  umap-learn not installed — skipping UMAP. "
                  "Install with: pip install umap-learn")

    # ------------------------------------------------------------------
    # 4. Level 1 — single-robot embeddings
    # ------------------------------------------------------------------
    print("\n[3/5] Level 1 — single-robot embedding visualization ...")
    for method in methods:
        print(f"  Running {method.upper()} on {total_points} points ...")
        X_2d = reduce_dimensions(
            data["embeddings"],
            method=method,
            perplexity=cfg["tsne_perplexity"],
            n_neighbors=cfg["umap_n_neighbors"],
            min_dist=cfg["umap_min_dist"],
            random_state=cfg["random_state"],
        )
        plot_level1(X_2d, data, method.upper(), save_dir, dpi=cfg["dpi"])

    # ------------------------------------------------------------------
    # 5. Level 2 — group embeddings
    # ------------------------------------------------------------------
    print(f"\n[4/5] Generating groups for level-2 ...")
    groups = generate_groups(
        num_robots=cfg["num_robots"],
        sizes=cfg["group_sizes"],
        use_structured=cfg["use_structured_groups"],
    )

    poolings: List[str] = []
    if cfg["group_pooling"] in ("mean", "both"):
        poolings.append("mean")
    if cfg["group_pooling"] in ("max", "both"):
        poolings.append("max")

    print(f"\n[5/5] Level 2 — group embedding visualization ...")
    for pooling in poolings:
        print(f"\n  Pooling: {pooling}")
        gdata = build_group_embeddings(data, groups, cfg["num_robots"], pooling)
        n_group_points = gdata["embeddings"].shape[0]
        print(f"  Group embeddings: {n_group_points} (snapshots × groups)")

        for method in methods:
            print(f"  Running {method.upper()} on {n_group_points} group points ...")
            # Adjust perplexity if needed (must be < n_samples / 3)
            perp = min(cfg["tsne_perplexity"], n_group_points // 4)
            perp = max(perp, 5)

            G_2d = reduce_dimensions(
                gdata["embeddings"],
                method=method,
                perplexity=perp,
                n_neighbors=min(cfg["umap_n_neighbors"], n_group_points - 1),
                min_dist=cfg["umap_min_dist"],
                random_state=cfg["random_state"],
            )
            plot_level2(G_2d, gdata, pooling, method.upper(), save_dir, dpi=cfg["dpi"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done! All figures saved to:")
    print(f"  {save_dir.resolve()}")
    print("=" * 60)

    # Print file listing
    for f in sorted(save_dir.glob("*.png")):
        print(f"  📊 {f.name}")


if __name__ == "__main__":
    main()
