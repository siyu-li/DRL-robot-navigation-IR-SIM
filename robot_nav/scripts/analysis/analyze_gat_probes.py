"""
GAT Embedding Analysis — Linear Probes + Embedding Decomposition.

This script answers two questions:
1. What information is linearly decodable from the GAT embedding?
   → Linear probes report R² / accuracy for each target feature.
2. Where does the information live — in the self-encoding or neighbor-aggregation?
   → We extract the 256-dim self_embed and 256-dim attn_out (before decode)
     and probe each half separately.
3. Does pooling destroy information for group embeddings?
   → We compare R² at the group level for mean-pool vs max-pool vs
     attention-weighted pool vs pairwise-difference features.

Usage:
    python -m robot_nav.scripts.analyze_gat_probes
"""

from __future__ import annotations

import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.groups.group_generator import generate_all_groups
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

# Suppress IRSim warnings
from loguru import logger as loguru_logger

loguru_logger.disable("irsim")


# =====================================================================
# Configuration
# =====================================================================
CONFIG = {
    # Model
    "model_name": "TD3-MARL-obstacle-14robots-partial-inactive_epoch210",
    "model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive",
    # Environment
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "num_robots": 14,
    "num_obstacles": 7,
    "state_dim": 11,
    "obstacle_state_dim": 4,
    # Data collection
    "n_snapshots": 800,
    "max_steps_per_episode": 200,
    # Probe settings
    "cv_folds": 5,
    # Group settings
    "group_sizes": [2, 3],
    "use_structured_groups": True,
    # Output
    "save_dir": "robot_nav/analysis/gat_probe_analysis",
    "dpi": 150,
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =====================================================================
# Custom embedding extractor — returns decomposed embeddings
# =====================================================================
def extract_decomposed_embeddings(
    actor: torch.nn.Module,
    robot_obs: np.ndarray,
    obstacle_obs: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract four levels of embedding from the frozen actor's attention module:

    1. full_embedding   (N, 512) — final att_embedding (after decode layers)
    2. self_embed       (N, 256) — robot's own encoded node features
    3. attn_out         (N, 256) — aggregated neighbor messages
    4. pre_decode       (N, 512) — [self_embed || attn_out] before decode

    This requires hooking into the attention forward pass.
    """
    robot_t = torch.tensor(robot_obs, dtype=torch.float32, device=device)
    obs_t = torch.tensor(obstacle_obs, dtype=torch.float32, device=device)

    if robot_t.dim() == 2:
        robot_t = robot_t.unsqueeze(0)
        obs_t = obs_t.unsqueeze(0)

    attn_module = actor.attention
    batch_size, n_robots, _ = robot_t.shape

    with torch.no_grad():
        # ---- Replicate the forward pass up to the decode step ----

        # Extract robot features (same indexing as forward())
        robot_feat = robot_t[:, :, 4:9]  # 5-dim: [dist/17, cos, sin, v, w]
        robot_position = robot_t[:, :, :2]
        robot_heading = robot_t[:, :, 2:4]
        robot_action = robot_t[:, :, 7:9]
        robot_goal = robot_t[:, :, -2:]

        # Encode robot node features → self_embed
        self_embed = attn_module.encode_robot_features(
            robot_feat.reshape(batch_size * n_robots, -1)
        )  # (B*N, 256)

        # Run the full attention forward to get attn_out
        (
            full_embedding,  # (B*N, 512)
            _, _, _, _, _,
            hard_weights_rr, hard_weights_ro, _,
        ) = attn_module(robot_t, obs_t)

        # To get attn_out, we note that:
        #   concat_embed = cat([self_embed, attn_out], dim=-1)  # (B*N, 512)
        #   x = leaky_relu(decode_1(concat_embed))
        #   att_embedding = leaky_relu(decode_2(x))
        #
        # We can't directly extract attn_out from the forward output,
        # so we reconstruct by running the encode again and using the
        # decode_1 layer's weight to invert... BUT that's lossy.
        #
        # Better approach: hook into the forward pass.
        # Since we already have self_embed, we can re-run the attention
        # without decode to get concat_embed.

        # Re-run obstacle encoding
        n_obs = obs_t.shape[1]
        obs_feat = torch.zeros(batch_size, n_obs, 5, device=device)
        obs_embed = attn_module.encode_obstacle_features(
            obs_feat.reshape(batch_size * n_obs, -1)
        ).view(batch_size, n_obs, attn_module.embedding_dim)

        robot_embed_3d = self_embed.view(batch_size, n_robots, attn_module.embedding_dim)

        # Build edges
        # We need edge features — recompute them
        pos_i = robot_position.unsqueeze(2)
        pos_j = robot_position.unsqueeze(1)
        heading_i = robot_heading.unsqueeze(2)
        heading_j = robot_heading.unsqueeze(1).expand(-1, n_robots, -1, -1)
        action_j = robot_action.unsqueeze(1).expand(-1, n_robots, -1, -1)
        goal_j = robot_goal.unsqueeze(1).expand(-1, n_robots, -1, -1)

        rel_vec_rr = pos_j - pos_i
        rel_dist_rr = torch.linalg.vector_norm(rel_vec_rr, dim=-1, keepdim=True) / 12
        dx_rr, dy_rr = rel_vec_rr[..., 0], rel_vec_rr[..., 1]
        angle_rr = torch.atan2(dy_rr, dx_rr) - torch.atan2(
            heading_i[..., 1], heading_i[..., 0]
        )
        angle_rr = (angle_rr + np.pi) % (2 * np.pi) - np.pi

        edge_features_rr = torch.cat(
            [
                rel_dist_rr,
                torch.cos(angle_rr).unsqueeze(-1),
                torch.sin(angle_rr).unsqueeze(-1),
                heading_j[..., 0:1],
                heading_j[..., 1:2],
                action_j,
            ],
            dim=-1,
        )

        obs_pos_j = obs_t[:, :, :2].unsqueeze(1)
        obs_heading = obs_t[:, :, 2:4]
        obs_heading_j = obs_heading.unsqueeze(1).expand(-1, n_robots, -1, -1)
        rel_vec_ro = obs_pos_j - pos_i
        rel_dist_ro = torch.linalg.vector_norm(rel_vec_ro, dim=-1, keepdim=True) / 12
        dx_ro, dy_ro = rel_vec_ro[..., 0], rel_vec_ro[..., 1]
        angle_ro = torch.atan2(dy_ro, dx_ro) - torch.atan2(
            heading_i[..., 1], heading_i[..., 0]
        )
        angle_ro = (angle_ro + np.pi) % (2 * np.pi) - np.pi

        edge_features_ro = torch.cat(
            [
                rel_dist_ro,
                torch.cos(angle_ro).unsqueeze(-1),
                torch.sin(angle_ro).unsqueeze(-1),
                obs_heading_j[..., 0:1],
                obs_heading_j[..., 1:2],
            ],
            dim=-1,
        )

        # Goal-relative polar
        goal_rel_vec = goal_j - pos_i
        goal_rel_dist = torch.linalg.vector_norm(goal_rel_vec, dim=-1, keepdim=True)
        goal_angle_global = torch.atan2(goal_rel_vec[..., 1], goal_rel_vec[..., 0])
        heading_angle = torch.atan2(heading_i[..., 1], heading_i[..., 0])
        goal_rel_angle = goal_angle_global - heading_angle
        goal_rel_angle = (goal_rel_angle + np.pi) % (2 * np.pi) - np.pi
        goal_polar_rr = torch.cat(
            [
                goal_rel_dist,
                torch.cos(goal_rel_angle).unsqueeze(-1),
                torch.sin(goal_rel_angle).unsqueeze(-1),
            ],
            dim=-1,
        )
        goal_polar_ro = torch.zeros(batch_size, n_robots, n_obs, 3, device=device)

        soft_edge_rr = torch.cat([edge_features_rr, goal_polar_rr], dim=-1)
        obs_action_zeros = torch.zeros(
            batch_size, n_robots, n_obs, 2, device=device
        )
        soft_edge_ro = torch.cat(
            [edge_features_ro, obs_action_zeros, goal_polar_ro], dim=-1
        )

        n_total = n_robots + n_obs

        edge_index, edge_attr, batch_ids = attn_module.build_edges_vectorized(
            hard_weights_rr, hard_weights_ro,
            soft_edge_rr, soft_edge_ro,
            n_robots, n_obs, device,
        )

        node_feats = torch.cat([robot_embed_3d, obs_embed], dim=1)
        node_feats_flat = node_feats.reshape(batch_size * n_total, -1)

        robot_indices = (
            torch.arange(batch_size, device=device).unsqueeze(1) * n_total
            + torch.arange(n_robots, device=device).unsqueeze(0)
        ).reshape(-1)
        q_robots_flat = attn_module.message_graph.q(node_feats_flat[robot_indices])

        total_edges = edge_index.shape[1]

        if total_edges > 0:
            tgt_global = edge_index[1]
            tgt_local_in_graph = tgt_global - batch_ids * n_total
            tgt_robot_space = batch_ids * n_robots + tgt_local_in_graph

            x_i = q_robots_flat[tgt_robot_space]
            k_edge = F.leaky_relu(attn_module.message_graph.k(edge_attr))
            v_edge = F.leaky_relu(attn_module.message_graph.v(edge_attr))
            from torch_geometric.utils import softmax as pyg_softmax

            attention_input = torch.cat([x_i, k_edge], dim=-1)
            scores = attn_module.message_graph.attn_score_layer(
                attention_input
            ).squeeze(-1)
            attn_weights = pyg_softmax(
                scores, tgt_robot_space, num_nodes=batch_size * n_robots
            )
            messages = v_edge * attn_weights.unsqueeze(-1)
            attn_out = torch.zeros(
                batch_size * n_robots, attn_module.embedding_dim, device=device
            )
            attn_out.index_add_(0, tgt_robot_space, messages)
        else:
            attn_out = torch.zeros(
                batch_size * n_robots, attn_module.embedding_dim, device=device
            )

        # Pre-decode = [self_embed || attn_out]
        pre_decode = torch.cat([self_embed, attn_out], dim=-1)

    return (
        full_embedding.cpu().numpy(),  # (N, 512)
        self_embed.cpu().numpy(),      # (N, 256)
        attn_out.cpu().numpy(),        # (N, 256)
        pre_decode.cpu().numpy(),      # (N, 512)
    )


# =====================================================================
# Data collection with decomposed embeddings
# =====================================================================
def collect_data(
    policy: TD3Obstacle,
    sim: MARL_SIM_OBSTACLE,
    device: torch.device,
    n_snapshots: int = 800,
    max_steps: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Collect per-robot embeddings (full + decomposed) and ground-truth targets.
    """
    all_full: List[np.ndarray] = []
    all_self: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []
    all_pre: List[np.ndarray] = []

    all_dtg: List[float] = []
    all_herr: List[float] = []
    all_nobs: List[float] = []
    all_nrob: List[float] = []
    all_rid: List[int] = []
    all_vel: List[List[float]] = []
    all_snap: List[int] = []

    (
        poses, distance, cos_val, sin_val, collision, goal, a, reward,
        positions, goal_positions, obstacle_states,
    ) = sim.reset(random_obstacles=True)

    N = sim.num_robots
    collected = 0
    step = 0

    while collected < n_snapshots:
        robot_state, _ = policy.prepare_state(
            poses, distance, cos_val, sin_val, collision, a, goal_positions
        )
        robot_obs = np.array(robot_state)

        # Decomposed extraction
        full_emb, self_emb, attn_emb, pre_emb = extract_decomposed_embeddings(
            policy.actor, robot_obs, obstacle_states, device
        )

        raw_action, combined_weights = policy.get_action(
            robot_obs, obstacle_states, add_noise=False
        )

        for i in range(N):
            px, py, theta = poses[i]
            gx, gy = goal_positions[i]

            dtg = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

            desired = np.arctan2(gy - py, gx - px)
            herr = abs(desired - theta)
            herr = min(herr, 2 * np.pi - herr)

            obs_pos = obstacle_states[:, :2]
            obs_d = np.sqrt(((obs_pos - np.array([px, py])) ** 2).sum(axis=1))
            nobs = float(obs_d.min()) if len(obs_d) > 0 else 999.0

            rdists = []
            for j in range(N):
                if j != i:
                    rdists.append(
                        np.sqrt(
                            (poses[j][0] - px) ** 2 + (poses[j][1] - py) ** 2
                        )
                    )
            nrob = min(rdists) if rdists else 0.0

            all_full.append(full_emb[i])
            all_self.append(self_emb[i])
            all_attn.append(attn_emb[i])
            all_pre.append(pre_emb[i])
            all_dtg.append(dtg)
            all_herr.append(herr)
            all_nobs.append(nobs)
            all_nrob.append(nrob)
            all_rid.append(i)
            all_vel.append(list(raw_action[i]))
            all_snap.append(collected)

        collected += 1
        step += 1

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

        if collected % 100 == 0:
            print(f"    {collected}/{n_snapshots} snapshots ...")

    return {
        "full_embedding": np.array(all_full),    # (S*N, 512)
        "self_embedding": np.array(all_self),     # (S*N, 256)
        "attn_embedding": np.array(all_attn),     # (S*N, 256)
        "pre_decode": np.array(all_pre),          # (S*N, 512)
        "dist_to_goal": np.array(all_dtg),
        "heading_error": np.array(all_herr),
        "nearest_obstacle": np.array(all_nobs),
        "nearest_robot": np.array(all_nrob),
        "robot_id": np.array(all_rid),
        "velocity": np.array(all_vel),            # (S*N, 2)
        "snapshot_id": np.array(all_snap),
    }


# =====================================================================
# Linear probes
# =====================================================================
def run_regression_probe(
    X: np.ndarray, y: np.ndarray, name: str, cv: int = 5
) -> float:
    """Cross-validated Ridge regression R²."""
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())


def run_classification_probe(
    X: np.ndarray, y: np.ndarray, name: str, cv: int = 5
) -> float:
    """Cross-validated logistic regression accuracy."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def probe_all_embeddings(
    data: Dict[str, np.ndarray], cv: int = 5
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Run linear probes on four embedding variants:
      full_embedding, self_embedding, attn_embedding, pre_decode
    against all regression and classification targets.

    Returns nested dict: results[embedding_name][target_name] = (mean, std)
    """
    embedding_keys = ["full_embedding", "self_embedding", "attn_embedding", "pre_decode"]
    regression_targets = {
        "dist_to_goal": data["dist_to_goal"],
        "heading_error": data["heading_error"],
        "nearest_obstacle": data["nearest_obstacle"],
        "nearest_robot": data["nearest_robot"],
        "linear_vel": data["velocity"][:, 0],
        "angular_vel": data["velocity"][:, 1],
    }
    classification_targets = {
        "robot_id": data["robot_id"],
    }

    results: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for emb_key in embedding_keys:
        X = data[emb_key]
        results[emb_key] = {}

        for tgt_name, y in regression_targets.items():
            mean_r2, std_r2 = run_regression_probe(X, y, tgt_name, cv)
            results[emb_key][f"R²_{tgt_name}"] = (mean_r2, std_r2)

        for tgt_name, y in classification_targets.items():
            if len(np.unique(y)) < 2:
                continue
            mean_acc, std_acc = run_classification_probe(X, y, tgt_name, cv)
            results[emb_key][f"Acc_{tgt_name}"] = (mean_acc, std_acc)

    return results


# =====================================================================
# Group-level probes — compare pooling strategies
# =====================================================================
def build_group_features(
    data: Dict[str, np.ndarray],
    groups: List[List[int]],
    n_robots: int,
    strategy: str = "mean",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build group-level features from robot embeddings using different strategies.

    Strategies:
      - "mean":       mean-pool embeddings
      - "max":        max-pool embeddings
      - "mean+std":   concatenate mean and std of member embeddings
      - "pairwise":   concatenate mean-pool + mean of pairwise differences
      - "concat_sorted": concatenate sorted (by dist_to_goal) member embeddings (fixed-size pad)

    Returns:
        X_group: (S * G, D_group)
        targets: dict of (S * G,) arrays for each target
    """
    emb = data["full_embedding"]
    n_total = emb.shape[0]
    n_snapshots = n_total // n_robots
    D = emb.shape[1]

    dtg_all = data["dist_to_goal"]
    herr_all = data["heading_error"]
    nobs_all = data["nearest_obstacle"]
    nrob_all = data["nearest_robot"]

    group_X_list: List[np.ndarray] = []
    group_dtg: List[float] = []
    group_herr: List[float] = []
    group_nobs: List[float] = []
    group_nrob: List[float] = []

    max_group_size = max(len(g) for g in groups)

    for t in range(n_snapshots):
        base = t * n_robots
        emb_t = emb[base: base + n_robots]
        dtg_t = dtg_all[base: base + n_robots]
        herr_t = herr_all[base: base + n_robots]
        nobs_t = nobs_all[base: base + n_robots]
        nrob_t = nrob_all[base: base + n_robots]

        for group in groups:
            idx = list(group)
            members = emb_t[idx]  # (G_size, D)

            if strategy == "mean":
                feat = members.mean(axis=0)
            elif strategy == "max":
                feat = members.max(axis=0)
            elif strategy == "mean+std":
                feat = np.concatenate([members.mean(axis=0), members.std(axis=0)])
            elif strategy == "pairwise":
                mean_feat = members.mean(axis=0)
                # Mean of absolute pairwise differences
                diffs = []
                for a_idx in range(len(idx)):
                    for b_idx in range(a_idx + 1, len(idx)):
                        diffs.append(np.abs(members[a_idx] - members[b_idx]))
                if diffs:
                    pair_feat = np.stack(diffs).mean(axis=0)
                else:
                    pair_feat = np.zeros(D)
                feat = np.concatenate([mean_feat, pair_feat])
            elif strategy == "concat_sorted":
                # Sort members by dist_to_goal, pad to max_group_size
                sort_idx = np.argsort(dtg_t[idx])
                sorted_members = members[sort_idx]
                padded = np.zeros((max_group_size, D))
                padded[: len(idx)] = sorted_members
                feat = padded.flatten()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            group_X_list.append(feat)
            # Aggregate targets with mean (consistent comparison)
            group_dtg.append(np.mean(dtg_t[idx]))
            group_herr.append(np.mean(herr_t[idx]))
            group_nobs.append(np.mean(nobs_t[idx]))
            group_nrob.append(np.mean(nrob_t[idx]))

    X_group = np.array(group_X_list)
    targets = {
        "dist_to_goal": np.array(group_dtg),
        "heading_error": np.array(group_herr),
        "nearest_obstacle": np.array(group_nobs),
        "nearest_robot": np.array(group_nrob),
    }
    return X_group, targets


def probe_group_strategies(
    data: Dict[str, np.ndarray],
    groups: List[List[int]],
    n_robots: int,
    cv: int = 5,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compare different group embedding strategies via linear probes.
    """
    strategies = ["mean", "max", "mean+std", "pairwise"]
    results: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for strat in strategies:
        print(f"    Strategy: {strat} ...")
        X_group, targets = build_group_features(data, groups, n_robots, strat)
        results[strat] = {}

        for tgt_name, y in targets.items():
            mean_r2, std_r2 = run_regression_probe(X_group, y, tgt_name, cv)
            results[strat][f"R²_{tgt_name}"] = (mean_r2, std_r2)

    return results


# =====================================================================
# Visualization helpers
# =====================================================================
def plot_probe_heatmap(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    title: str,
    save_path: Path,
    dpi: int = 150,
):
    """Plot a heatmap of probe R² / Accuracy values."""
    row_labels = list(results.keys())
    # Collect all target names across all embedding types
    all_targets = []
    for targets in results.values():
        for t in targets:
            if t not in all_targets:
                all_targets.append(t)
    col_labels = all_targets

    matrix = np.zeros((len(row_labels), len(col_labels)))
    annot = [['' for _ in col_labels] for _ in row_labels]

    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            if col in results[row]:
                mean, std = results[row][col]
                matrix[i, j] = mean
                annot[i][j] = f"{mean:.3f}\n±{std:.3f}"
            else:
                matrix[i, j] = np.nan
                annot[i][j] = "N/A"

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.8), max(4, len(row_labels) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="R² / Accuracy")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([c.replace("R²_", "").replace("Acc_", "Acc:") for c in col_labels],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=7,
                    color="black" if matrix[i, j] > 0.5 else "white")

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path}")


def plot_bar_comparison(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    title: str,
    save_path: Path,
    dpi: int = 150,
):
    """Bar chart comparing strategies for group R²."""
    strategies = list(results.keys())
    targets = list(results[strategies[0]].keys())

    n_strat = len(strategies)
    n_tgt = len(targets)
    x = np.arange(n_tgt)
    width = 0.8 / n_strat

    fig, ax = plt.subplots(figsize=(max(8, n_tgt * 2), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n_strat))

    for i, strat in enumerate(strategies):
        means = [results[strat].get(t, (0, 0))[0] for t in targets]
        stds = [results[strat].get(t, (0, 0))[1] for t in targets]
        ax.bar(x + i * width, means, width, yerr=stds,
               label=strat, color=colors[i], capsize=3)

    ax.set_xlabel("Target")
    ax.set_ylabel("R²")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (n_strat - 1) / 2)
    ax.set_xticklabels([t.replace("R²_", "") for t in targets], rotation=30, ha="right")
    ax.legend(title="Strategy", fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path}")


# =====================================================================
# Main
# =====================================================================
def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAT Embedding Analysis — Linear Probes + Decomposition")
    print("=" * 70)

    # ---- Load ----
    print("\n[1/6] Loading environment and policy ...")
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
    print(f"  Loaded: {cfg['model_name']}")

    # ---- Collect ----
    print(f"\n[2/6] Collecting {cfg['n_snapshots']} snapshots with decomposed embeddings ...")
    data = collect_data(
        policy, sim, device,
        n_snapshots=cfg["n_snapshots"],
        max_steps=cfg["max_steps_per_episode"],
    )
    n_points = data["full_embedding"].shape[0]
    print(f"  Total data points: {n_points}")
    print(f"  Embedding dims: full={data['full_embedding'].shape[1]}, "
          f"self={data['self_embedding'].shape[1]}, "
          f"attn={data['attn_embedding'].shape[1]}")

    # ---- Level 1: Per-robot probes (decomposed) ----
    print(f"\n[3/6] Running linear probes on 4 embedding variants ...")
    probe_results = probe_all_embeddings(data, cv=cfg["cv_folds"])

    # Print results table
    print("\n" + "=" * 70)
    print("LEVEL 1 — Per-Robot Linear Probe Results")
    print("=" * 70)
    header = f"{'Embedding':<20}"
    first_key = list(probe_results.keys())[0]
    targets = list(probe_results[first_key].keys())
    for t in targets:
        short = t.replace("R²_", "").replace("Acc_", "Acc:")
        header += f" {short:>14}"
    print(header)
    print("-" * len(header))

    for emb_name, tgt_dict in probe_results.items():
        row = f"{emb_name:<20}"
        for t in targets:
            if t in tgt_dict:
                mean, std = tgt_dict[t]
                row += f" {mean:>10.4f}±{std:.2f}"
            else:
                row += f" {'N/A':>14}"
        print(row)

    # Plot heatmap
    plot_probe_heatmap(
        probe_results,
        "Level 1: Per-Robot Probe — R² / Accuracy by Embedding",
        save_dir / "level1_probe_heatmap.png",
        dpi=cfg["dpi"],
    )

    # ---- Interpret Level 1 ----
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    full_dtg = probe_results["full_embedding"].get("R²_dist_to_goal", (0, 0))[0]
    self_dtg = probe_results["self_embedding"].get("R²_dist_to_goal", (0, 0))[0]
    attn_dtg = probe_results["attn_embedding"].get("R²_dist_to_goal", (0, 0))[0]

    full_nrob = probe_results["full_embedding"].get("R²_nearest_robot", (0, 0))[0]
    self_nrob = probe_results["self_embedding"].get("R²_nearest_robot", (0, 0))[0]
    attn_nrob = probe_results["attn_embedding"].get("R²_nearest_robot", (0, 0))[0]

    full_nobs = probe_results["full_embedding"].get("R²_nearest_obstacle", (0, 0))[0]
    attn_nobs = probe_results["attn_embedding"].get("R²_nearest_obstacle", (0, 0))[0]

    full_rid = probe_results["full_embedding"].get("Acc_robot_id", (0, 0))[0]

    print(f"  dist_to_goal:      self={self_dtg:.3f}  attn={attn_dtg:.3f}  full={full_dtg:.3f}")
    if self_dtg > attn_dtg + 0.1:
        print("    → Primarily encoded in self_embed (ego-centric node features)")
    elif attn_dtg > self_dtg + 0.1:
        print("    → Primarily encoded in attn_out (neighbor messages)")

    print(f"  nearest_robot:     self={self_nrob:.3f}  attn={attn_nrob:.3f}  full={full_nrob:.3f}")
    if attn_nrob > self_nrob + 0.05:
        print("    → Better in attn_out — spatial info comes through message passing")
    elif full_nrob < 0.3:
        print("    → Poorly encoded everywhere — GAT consumes proximity for avoidance,")
        print("      doesn't retain it. Need explicit features for group tasks.")

    print(f"  nearest_obstacle:  attn={attn_nobs:.3f}  full={full_nobs:.3f}")
    if full_nobs < 0.3:
        print("    → Same: consumed for collision avoidance, not retained.")

    print(f"  robot_id accuracy: {full_rid:.3f}")
    if full_rid > 0.5:
        print("    → ⚠️  Embedding is partially identity-specific.")
        print("      Mean-pooling may mix incompatible representations.")
    else:
        print("    → ✅ Embedding is NOT identity-specific (good for pooling).")

    # ---- Level 2: Group probes ----
    print(f"\n[4/6] Generating groups ...")
    m = 4 if cfg["num_robots"] > 6 else 3
    all_groups = generate_all_groups(m=m, n=cfg["num_robots"], use_complement=True)
    groups = [g for g in all_groups if len(g) in cfg["group_sizes"]]
    print(f"  {len(groups)} groups (sizes {cfg['group_sizes']})")

    print(f"\n[5/6] Running group-level probes for different pooling strategies ...")
    group_results = probe_group_strategies(
        data, groups, cfg["num_robots"], cv=cfg["cv_folds"]
    )

    # Print results
    print("\n" + "=" * 70)
    print("LEVEL 2 — Group Embedding Probe (R²) by Pooling Strategy")
    print("=" * 70)
    strats = list(group_results.keys())
    gtargets = list(group_results[strats[0]].keys())

    header = f"{'Strategy':<15}"
    for t in gtargets:
        short = t.replace("R²_", "")
        header += f" {short:>16}"
    print(header)
    print("-" * len(header))

    for strat in strats:
        row = f"{strat:<15}"
        for t in gtargets:
            mean, std = group_results[strat].get(t, (0, 0))
            row += f" {mean:>12.4f}±{std:.2f}"
        print(row)

    # Plot bar chart
    plot_bar_comparison(
        group_results,
        "Level 2: Group Embedding R² by Pooling Strategy",
        save_dir / "level2_group_strategy_comparison.png",
        dpi=cfg["dpi"],
    )

    # ---- Summary + Recommendations ----
    print(f"\n[6/6] Generating summary ...")
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find best group strategy for dist_to_goal
    best_strat = max(strats, key=lambda s: group_results[s].get("R²_dist_to_goal", (0, 0))[0])
    best_r2 = group_results[best_strat]["R²_dist_to_goal"][0]
    mean_r2 = group_results["mean"]["R²_dist_to_goal"][0]

    if best_strat != "mean":
        improvement = best_r2 - mean_r2
        print(f"  1. '{best_strat}' pooling outperforms mean-pool by +{improvement:.3f} R²")
        print(f"     → Consider using {best_strat} in your group switcher/coupling.")

    meanstd_r2 = group_results.get("mean+std", {}).get("R²_dist_to_goal", (0, 0))[0]
    if meanstd_r2 > mean_r2 + 0.02:
        print(f"  2. mean+std ({meanstd_r2:.3f}) > mean ({mean_r2:.3f}):")
        print(f"     → Variance within the group carries information!")
        print(f"     → Add std-pooling features to your RLFeatureBuilder.")

    pair_r2 = group_results.get("pairwise", {}).get("R²_dist_to_goal", (0, 0))[0]
    if pair_r2 > mean_r2 + 0.02:
        print(f"  3. pairwise ({pair_r2:.3f}) > mean ({mean_r2:.3f}):")
        print(f"     → Pairwise differences between members carry group-level info.")
        print(f"     → Consider a pairwise interaction layer in the group network.")

    if full_nrob < 0.3 and full_nobs < 0.3:
        print(f"  4. nearest_robot (R²={full_nrob:.3f}) and nearest_obstacle (R²={full_nobs:.3f})")
        print(f"     are poorly encoded in the full embedding.")
        print(f"     → Append explicit spatial scalars (min clearance, nearest-robot dist)")
        print(f"       as additional input features to your switcher and coupling networks.")

    print(f"\n  All figures saved to: {save_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
