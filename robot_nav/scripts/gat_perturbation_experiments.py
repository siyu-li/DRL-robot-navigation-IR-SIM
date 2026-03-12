"""
GAT Perturbation Experiments — Neighbor Removal Sensitivity Analysis.

Four experiments probing the internal representations of a trained
Graph Attention Network for multi-robot navigation:

Experiment 1: Single-neighbor removal sensitivity
   For each robot i, remove each neighbor j one at a time and compute
   Δe_i = e_i(all) - e_i(without j). Train linear and nonlinear probes
   on Δe_i to predict geometric relationship (distance, angle, velocity,
   entity type). Tests whether spatial information about each neighbor is
   encoded as that neighbor's contribution to the aggregate embedding.

Experiment 2: Superposition test
   Remove neighbors j and k individually (Δe^j, Δe^k) and together
   (Δe^{jk}). Check whether Δe^{jk} ≈ Δe^j + Δe^k. Additive ⟹
   neighbors encoded independently; non-additive ⟹ interaction effects
   from attention re-normalization.

Experiment 3: Behavioral relevance of deltas
   For the same perturbations, compute action deltas Δa_i and correlate
   with geometric features and embedding deltas. Tests whether the
   relational information is tightly coupled to the policy output.

Experiment 4: Attention weight inspection
   Extract learned attention weights and plot against distance, relative
   heading, and entity type. Directly reveals the implicit relational
   encoding without perturbation.

Usage:
    python -m robot_nav.scripts.gat_perturbation_experiments
"""

from __future__ import annotations

import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
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
    "n_snapshots": 500,
    "max_steps_per_episode": 200,
    # Probes
    "cv_folds": 5,
    # Superposition
    "max_pairs_per_robot": 15,
    # Output
    "save_dir": "robot_nav/analysis/perturbation_experiments",
    "dpi": 150,
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Seed
    "seed": 42,
}


# =====================================================================
# Section 1 — Core Perturbation Engine
# =====================================================================

@torch.no_grad()
def precompute_attention_intermediates(
    attn_module: torch.nn.Module,
    robot_t: torch.Tensor,
    obs_t: torch.Tensor,
    device: torch.device,
) -> Dict:
    """
    Run the attention module's forward pass and extract all intermediate
    quantities needed for efficient perturbation analysis.

    This replicates the forward pass through:
      node encoding → edge features → hard attention → edge construction →
      per-edge score/value computation → baseline softmax → baseline
      attn_out → decode.

    Everything except the final softmax/aggregation/decode is reusable
    across all perturbations for the same snapshot.

    Args:
        attn_module: The ``AttentionObstacleOptimized`` module (frozen).
        robot_t: Robot observations, shape ``(N, 11)`` or ``(1, N, 11)``.
        obs_t: Obstacle observations, shape ``(N_obs, 4)`` or ``(1, N_obs, 4)``.
        device: Torch device.

    Returns:
        Dict with all intermediates needed for perturbation analysis.
    """
    if robot_t.dim() == 2:
        robot_t = robot_t.unsqueeze(0)
    if obs_t.dim() == 2:
        obs_t = obs_t.unsqueeze(0)

    B, N, _ = robot_t.shape
    _, N_obs, _ = obs_t.shape
    n_total = N + N_obs

    # ---- Extract raw features ----
    robot_feat = robot_t[:, :, 4:9]
    robot_position = robot_t[:, :, :2]
    robot_heading = robot_t[:, :, 2:4]
    robot_action = robot_t[:, :, 7:9]
    robot_goal = robot_t[:, :, -2:]

    obs_position = obs_t[:, :, :2]
    obs_heading_raw = obs_t[:, :, 2:4]
    obs_feat = torch.zeros(B, N_obs, 5, device=device)

    # ---- Encode node features ----
    robot_embed = attn_module.encode_robot_features(
        robot_feat.reshape(B * N, -1)
    ).view(B, N, attn_module.embedding_dim)

    obs_embed = attn_module.encode_obstacle_features(
        obs_feat.reshape(B * N_obs, -1)
    ).view(B, N_obs, attn_module.embedding_dim)

    # ---- Robot-robot edge features (vectorized) ----
    pos_i = robot_position.unsqueeze(2)
    pos_j = robot_position.unsqueeze(1)
    heading_i = robot_heading.unsqueeze(2)
    heading_j = robot_heading.unsqueeze(1).expand(-1, N, -1, -1)
    action_j = robot_action.unsqueeze(1).expand(-1, N, -1, -1)
    goal_j = robot_goal.unsqueeze(1).expand(-1, N, -1, -1)

    rel_vec_rr = pos_j - pos_i
    rel_dist_rr = torch.linalg.vector_norm(rel_vec_rr, dim=-1, keepdim=True) / 12
    dx_rr, dy_rr = rel_vec_rr[..., 0], rel_vec_rr[..., 1]
    angle_rr = torch.atan2(dy_rr, dx_rr) - torch.atan2(
        heading_i[..., 1], heading_i[..., 0]
    )
    angle_rr = (angle_rr + np.pi) % (2 * np.pi) - np.pi

    edge_features_rr = torch.cat([
        rel_dist_rr,
        torch.cos(angle_rr).unsqueeze(-1),
        torch.sin(angle_rr).unsqueeze(-1),
        heading_j[..., 0:1],
        heading_j[..., 1:2],
        action_j,
    ], dim=-1)

    # ---- Robot-obstacle edge features ----
    obs_pos_j = obs_position.unsqueeze(1)
    obs_heading_j = obs_heading_raw.unsqueeze(1).expand(-1, N, -1, -1)

    rel_vec_ro = obs_pos_j - pos_i
    rel_dist_ro = torch.linalg.vector_norm(rel_vec_ro, dim=-1, keepdim=True) / 12
    dx_ro, dy_ro = rel_vec_ro[..., 0], rel_vec_ro[..., 1]
    angle_ro = torch.atan2(dy_ro, dx_ro) - torch.atan2(
        heading_i[..., 1], heading_i[..., 0]
    )
    angle_ro = (angle_ro + np.pi) % (2 * np.pi) - np.pi

    edge_features_ro = torch.cat([
        rel_dist_ro,
        torch.cos(angle_ro).unsqueeze(-1),
        torch.sin(angle_ro).unsqueeze(-1),
        obs_heading_j[..., 0:1],
        obs_heading_j[..., 1:2],
    ], dim=-1)

    # ---- Hard attention (robot-robot) ----
    h_i_rr = robot_embed.unsqueeze(2).expand(-1, -1, N, -1)
    hard_input_rr = torch.cat([h_i_rr, edge_features_rr], dim=-1)
    hard_input_rr = hard_input_rr.reshape(B * N, N, -1)

    h_hard_rr = attn_module.hard_mlp(hard_input_rr)
    hard_logits_rr = attn_module.hard_encoding(h_hard_rr)
    hard_weights_rr = F.gumbel_softmax(
        hard_logits_rr, hard=False, tau=0.2, dim=-1
    )[..., 1]
    hard_weights_rr = hard_weights_rr.view(B, N, N)

    # ---- Hard attention (robot-obstacle) ----
    h_i_ro = robot_embed.unsqueeze(2).expand(-1, -1, N_obs, -1)
    hard_input_ro = torch.cat([h_i_ro, edge_features_ro], dim=-1)
    hard_input_ro = hard_input_ro.reshape(B * N, N_obs, -1)

    h_hard_ro = attn_module.hard_mlp_obs(hard_input_ro)
    hard_logits_ro = attn_module.hard_encoding_obs(h_hard_ro)
    hard_weights_ro = F.gumbel_softmax(
        hard_logits_ro, hard=False, tau=0.2, dim=-1
    )[..., 1]
    hard_weights_ro = hard_weights_ro.view(B, N, N_obs)

    # ---- Goal-relative polar features ----
    goal_rel_vec = goal_j - pos_i
    goal_rel_dist = torch.linalg.vector_norm(goal_rel_vec, dim=-1, keepdim=True)
    goal_angle_global = torch.atan2(goal_rel_vec[..., 1], goal_rel_vec[..., 0])
    heading_angle = torch.atan2(heading_i[..., 1], heading_i[..., 0])
    goal_rel_angle = goal_angle_global - heading_angle
    goal_rel_angle = (goal_rel_angle + np.pi) % (2 * np.pi) - np.pi
    goal_polar_rr = torch.cat([
        goal_rel_dist,
        torch.cos(goal_rel_angle).unsqueeze(-1),
        torch.sin(goal_rel_angle).unsqueeze(-1),
    ], dim=-1)
    goal_polar_ro = torch.zeros(B, N, N_obs, 3, device=device)

    # ---- Soft edge features (10-dim) ----
    soft_edge_rr = torch.cat([edge_features_rr, goal_polar_rr], dim=-1)
    obs_action_zeros = torch.zeros(B, N, N_obs, 2, device=device)
    soft_edge_ro = torch.cat([
        edge_features_ro, obs_action_zeros, goal_polar_ro
    ], dim=-1)

    # ---- Build edges ----
    edge_index, edge_attr, batch_ids = attn_module.build_edges_vectorized(
        hard_weights_rr, hard_weights_ro,
        soft_edge_rr, soft_edge_ro,
        N, N_obs, device,
    )

    # ---- Per-edge score and value computation ----
    node_feats = torch.cat([robot_embed, obs_embed], dim=1)  # (B, n_total, D)
    node_feats_flat = node_feats.reshape(B * n_total, -1)

    robot_indices = (
        torch.arange(B, device=device).unsqueeze(1) * n_total
        + torch.arange(N, device=device).unsqueeze(0)
    ).reshape(-1)
    q_robots_flat = attn_module.message_graph.q(node_feats_flat[robot_indices])

    total_edges = edge_index.shape[1]

    if total_edges > 0:
        tgt_global = edge_index[1]
        tgt_local = tgt_global - batch_ids * n_total
        tgt_robot_space = batch_ids * N + tgt_local

        x_i = q_robots_flat[tgt_robot_space]
        k_edge = F.leaky_relu(attn_module.message_graph.k(edge_attr))
        v_edge = F.leaky_relu(attn_module.message_graph.v(edge_attr))

        attention_input = torch.cat([x_i, k_edge], dim=-1)
        scores = attn_module.message_graph.attn_score_layer(
            attention_input
        ).squeeze(-1)

        from torch_geometric.utils import softmax as pyg_softmax
        attn_weights = pyg_softmax(scores, tgt_robot_space, num_nodes=B * N)

        # Baseline attn_out
        messages = v_edge * attn_weights.unsqueeze(-1)
        baseline_attn_out = torch.zeros(
            B * N, attn_module.embedding_dim, device=device
        )
        baseline_attn_out.index_add_(0, tgt_robot_space, messages)
    else:
        scores = torch.zeros(0, device=device)
        v_edge = torch.zeros(0, attn_module.embedding_dim, device=device)
        attn_weights = torch.zeros(0, device=device)
        tgt_robot_space = torch.zeros(0, dtype=torch.long, device=device)
        baseline_attn_out = torch.zeros(
            B * N, attn_module.embedding_dim, device=device
        )

    # ---- Baseline decode ----
    self_embed = robot_embed.reshape(B * N, -1)
    concat = torch.cat([self_embed, baseline_attn_out], dim=-1)
    x = F.leaky_relu(attn_module.decode_1(concat))
    baseline_embedding = F.leaky_relu(attn_module.decode_2(x))

    return {
        "robot_embed": robot_embed,          # (B, N, 256)
        "obs_embed": obs_embed,              # (B, N_obs, 256)
        "self_embed": self_embed,            # (B*N, 256)
        "baseline_attn_out": baseline_attn_out,  # (B*N, 256)
        "baseline_embedding": baseline_embedding,  # (B*N, 512)
        "edge_index": edge_index,            # (2, E)
        "scores": scores,                    # (E,)
        "v_edge": v_edge,                    # (E, 256)
        "attn_weights": attn_weights,        # (E,)
        "tgt_robot_space": tgt_robot_space,  # (E,)
        "hard_weights_rr": hard_weights_rr,  # (B, N, N)
        "hard_weights_ro": hard_weights_ro,  # (B, N, N_obs)
        "robot_position": robot_position,    # (B, N, 2)
        "robot_heading": robot_heading,      # (B, N, 2)
        "robot_velocity": robot_action,      # (B, N, 2) = [lin_vel, ang_vel]
        "obs_position": obs_position,        # (B, N_obs, 2)
        "obs_heading": obs_heading_raw,      # (B, N_obs, 2)
        "n_robots": N,
        "n_obs": N_obs,
    }


@torch.no_grad()
def compute_single_removal_deltas(
    intermediates: Dict,
    attn_module: torch.nn.Module,
    policy_head: torch.nn.Module,
    device: torch.device,
) -> List[Dict]:
    """
    For every robot in the snapshot, compute embedding and action deltas
    when each active neighbor is individually removed.

    Uses efficient masked softmax: all K perturbations for one robot are
    computed in a single batched operation (K × K softmax + matmul).

    Args:
        intermediates: Output of ``precompute_attention_intermediates``.
        attn_module: The attention module (for decode layers).
        policy_head: The actor's policy head (for action computation).
        device: Torch device.

    Returns:
        List of dicts, one per (robot_i, removed_neighbor) pair, containing
        geometric features, attention weight, and delta vectors.
    """
    N = intermediates["n_robots"]
    N_obs = intermediates["n_obs"]
    edge_index = intermediates["edge_index"]
    scores = intermediates["scores"]
    v_edge = intermediates["v_edge"]
    attn_weights_all = intermediates["attn_weights"]
    self_embed = intermediates["self_embed"]
    baseline_embedding = intermediates["baseline_embedding"]
    baseline_attn_out = intermediates["baseline_attn_out"]
    embed_dim = attn_module.embedding_dim

    # Precompute baseline actions for all robots
    baseline_actions = policy_head(baseline_embedding)  # (N, 2)

    # Raw state for geometric features (batch=0)
    robot_pos = intermediates["robot_position"][0]   # (N, 2)
    robot_head = intermediates["robot_heading"][0]    # (N, 2)
    robot_vel = intermediates["robot_velocity"][0]    # (N, 2)
    obs_pos = intermediates["obs_position"][0]        # (N_obs, 2)
    obs_head = intermediates["obs_heading"][0]        # (N_obs, 2)

    records: List[Dict] = []

    for i in range(N):
        # Find edges targeting robot i (batch_size=1)
        edge_mask = (edge_index[1] == i)
        edge_indices = torch.where(edge_mask)[0]
        K = edge_indices.shape[0]

        if K == 0:
            continue

        scores_i = scores[edge_indices]       # (K,)
        values_i = v_edge[edge_indices]       # (K, D)
        sources_i = edge_index[0, edge_indices]  # (K,) global source indices
        weights_i = attn_weights_all[edge_indices]  # (K,) baseline weights

        # ---- All K perturbations via masked softmax ----
        if K == 1:
            # Removing the only neighbor → attn_out = 0
            perturbed_attn_outs = torch.zeros(1, embed_dim, device=device)
        else:
            score_matrix = scores_i.unsqueeze(0).expand(K, -1).clone()
            # Mask diagonal: each row k masks out edge k
            score_matrix[
                torch.arange(K, device=device),
                torch.arange(K, device=device),
            ] = float("-inf")
            perturbed_weights = F.softmax(score_matrix, dim=1)
            perturbed_weights = torch.nan_to_num(perturbed_weights, nan=0.0)
            perturbed_attn_outs = torch.matmul(
                perturbed_weights, values_i
            )  # (K, D)

        # ---- Decode ----
        self_i = self_embed[i].unsqueeze(0).expand(K, -1)
        concat = torch.cat([self_i, perturbed_attn_outs], dim=1)  # (K, 2D)
        x = F.leaky_relu(attn_module.decode_1(concat))
        perturbed_embeddings = F.leaky_relu(attn_module.decode_2(x))

        # ---- Actions ----
        perturbed_actions = policy_head(perturbed_embeddings)

        # ---- Deltas ----
        delta_attn_out = perturbed_attn_outs - baseline_attn_out[i].unsqueeze(0)
        delta_embedding = perturbed_embeddings - baseline_embedding[i].unsqueeze(0)
        delta_action = perturbed_actions - baseline_actions[i].unsqueeze(0)

        # ---- Geometric features per neighbor ----
        pos_i = robot_pos[i]
        heading_angle_i = torch.atan2(robot_head[i, 1], robot_head[i, 0])

        for k in range(K):
            src = sources_i[k].item()
            if src < N:
                # Robot neighbor
                p_j = robot_pos[src]
                h_j = robot_head[src]
                v_j = robot_vel[src]
                is_robot = True
                neighbor_idx = src
            else:
                # Obstacle neighbor
                obs_idx = src - N
                p_j = obs_pos[obs_idx]
                h_j = obs_head[obs_idx]
                v_j = torch.zeros(2, device=device)
                is_robot = False
                neighbor_idx = obs_idx

            rel_vec = p_j - pos_i
            dist = torch.linalg.norm(rel_vec).item()
            angle_to_j = torch.atan2(rel_vec[1], rel_vec[0]).item()
            rel_angle = angle_to_j - heading_angle_i.item()
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

            heading_angle_j = torch.atan2(h_j[1], h_j[0]).item()
            rel_heading = heading_angle_j - heading_angle_i.item()
            rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi

            records.append({
                "robot_i": i,
                "neighbor_j": neighbor_idx,
                "neighbor_type": 1 if is_robot else 0,  # 1=robot, 0=obstacle
                "distance": dist,
                "relative_angle": rel_angle,
                "rel_heading_cos": np.cos(rel_heading),
                "rel_heading_sin": np.sin(rel_heading),
                "rel_velocity_lin": v_j[0].item(),
                "rel_velocity_ang": v_j[1].item(),
                "attn_weight": weights_i[k].item(),
                "delta_attn_out": delta_attn_out[k].cpu().numpy(),
                "delta_embedding": delta_embedding[k].cpu().numpy(),
                "delta_action": delta_action[k].cpu().numpy(),
            })

    return records


@torch.no_grad()
def compute_pair_removal_deltas(
    intermediates: Dict,
    attn_module: torch.nn.Module,
    device: torch.device,
    max_pairs: int = 15,
) -> List[Dict]:
    """
    For every robot, compute embedding deltas when pairs of neighbors are
    removed together, and compare to the sum of individual deltas.

    Superposition is tested at the ``attn_out`` level (256-dim, before
    nonlinear decode) because the decode layers' nonlinearity will always
    break strict additivity.

    Args:
        intermediates: Output of ``precompute_attention_intermediates``.
        attn_module: The attention module (for embed_dim).
        device: Torch device.
        max_pairs: Maximum number of pairs to sample per robot.

    Returns:
        List of dicts with superposition metrics for each (robot, pair).
    """
    N = intermediates["n_robots"]
    N_obs = intermediates["n_obs"]
    edge_index = intermediates["edge_index"]
    scores = intermediates["scores"]
    v_edge = intermediates["v_edge"]
    baseline_attn_out = intermediates["baseline_attn_out"]
    embed_dim = attn_module.embedding_dim

    robot_pos = intermediates["robot_position"][0]
    obs_pos = intermediates["obs_position"][0]

    pair_records: List[Dict] = []

    for i in range(N):
        edge_mask = (edge_index[1] == i)
        edge_indices = torch.where(edge_mask)[0]
        K = edge_indices.shape[0]

        if K < 2:
            continue

        scores_i = scores[edge_indices]
        values_i = v_edge[edge_indices]
        sources_i = edge_index[0, edge_indices]
        baseline_attn_i = baseline_attn_out[i]

        # Sample pairs
        all_pairs = list(combinations(range(K), 2))
        if len(all_pairs) > max_pairs:
            all_pairs = random.sample(all_pairs, max_pairs)

        for j_local, k_local in all_pairs:
            # ---- Single removal: j ----
            mask_j = scores_i.clone()
            mask_j[j_local] = float("-inf")
            if K == 1:
                attn_j = torch.zeros(embed_dim, device=device)
            else:
                w_j = F.softmax(mask_j, dim=0)
                w_j = torch.nan_to_num(w_j, nan=0.0)
                attn_j = (w_j.unsqueeze(-1) * values_i).sum(dim=0)
            delta_j = attn_j - baseline_attn_i

            # ---- Single removal: k ----
            mask_k = scores_i.clone()
            mask_k[k_local] = float("-inf")
            if K == 1:
                attn_k = torch.zeros(embed_dim, device=device)
            else:
                w_k = F.softmax(mask_k, dim=0)
                w_k = torch.nan_to_num(w_k, nan=0.0)
                attn_k = (w_k.unsqueeze(-1) * values_i).sum(dim=0)
            delta_k = attn_k - baseline_attn_i

            # ---- Pair removal: j and k ----
            mask_jk = scores_i.clone()
            mask_jk[j_local] = float("-inf")
            mask_jk[k_local] = float("-inf")
            remaining = K - 2
            if remaining == 0:
                attn_jk = torch.zeros(embed_dim, device=device)
            else:
                w_jk = F.softmax(mask_jk, dim=0)
                w_jk = torch.nan_to_num(w_jk, nan=0.0)
                attn_jk = (w_jk.unsqueeze(-1) * values_i).sum(dim=0)
            delta_jk = attn_jk - baseline_attn_i

            # ---- Superposition check ----
            delta_sum = delta_j + delta_k
            residual = delta_jk - delta_sum

            delta_jk_norm = torch.linalg.norm(delta_jk).item()
            delta_sum_norm = torch.linalg.norm(delta_sum).item()
            residual_norm = torch.linalg.norm(residual).item()

            # Cosine similarity (handle zero vectors)
            if delta_jk_norm > 1e-8 and delta_sum_norm > 1e-8:
                cos_sim = F.cosine_similarity(
                    delta_jk.unsqueeze(0), delta_sum.unsqueeze(0)
                ).item()
            else:
                cos_sim = 1.0 if delta_jk_norm < 1e-8 and delta_sum_norm < 1e-8 else 0.0

            relative_residual = (
                residual_norm / delta_jk_norm if delta_jk_norm > 1e-8 else 0.0
            )

            # ---- Distances ----
            pos_i = robot_pos[i]
            src_j = sources_i[j_local].item()
            src_k = sources_i[k_local].item()

            def _get_pos(src_idx):
                if src_idx < N:
                    return robot_pos[src_idx]
                return obs_pos[src_idx - N]

            p_j = _get_pos(src_j)
            p_k = _get_pos(src_k)

            pair_records.append({
                "robot_i": i,
                "delta_jk_norm": delta_jk_norm,
                "delta_sum_norm": delta_sum_norm,
                "residual_norm": residual_norm,
                "relative_residual": relative_residual,
                "cosine_sim": cos_sim,
                "dist_ij": torch.linalg.norm(p_j - pos_i).item(),
                "dist_ik": torch.linalg.norm(p_k - pos_i).item(),
                "dist_jk": torch.linalg.norm(p_k - p_j).item(),
            })

    return pair_records


# =====================================================================
# Section 2 — Data Collection
# =====================================================================

def collect_perturbation_data(
    policy: TD3Obstacle,
    sim: MARL_SIM_OBSTACLE,
    device: torch.device,
    n_snapshots: int = 500,
    max_steps: int = 200,
    max_pairs: int = 15,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run the simulation, collect snapshots, and compute perturbation data
    for all four experiments.

    Returns three dicts:
        single_data:  arrays for single-neighbor removal (experiments 1, 3, 4)
        pair_data:    arrays for pair removal (experiment 2)
        attn_data:    arrays for attention weight analysis (experiment 4)
    """
    actor = policy.actor
    attn_module = actor.attention
    policy_head = actor.policy_head

    # Accumulation lists — single removal
    s_keys = [
        "robot_i", "neighbor_j", "neighbor_type", "distance",
        "relative_angle", "rel_heading_cos", "rel_heading_sin",
        "rel_velocity_lin", "rel_velocity_ang", "attn_weight",
    ]
    s_scalar = {k: [] for k in s_keys}
    s_delta_attn = []
    s_delta_emb = []
    s_delta_act = []
    s_snapshot = []

    # Accumulation lists — pair removal
    p_keys = [
        "robot_i", "delta_jk_norm", "delta_sum_norm", "residual_norm",
        "relative_residual", "cosine_sim", "dist_ij", "dist_ik", "dist_jk",
    ]
    p_scalar = {k: [] for k in p_keys}
    p_snapshot = []

    # Accumulation lists — attention weights (experiment 4)
    a_keys = [
        "robot_i", "source_idx", "is_robot", "distance",
        "relative_angle", "rel_heading", "attn_weight",
    ]
    a_scalar = {k: [] for k in a_keys}
    a_snapshot = []

    # ---- Simulation loop ----
    (
        poses, distance, cos_val, sin_val, collision, goal, a, reward,
        positions, goal_positions, obstacle_states,
    ) = sim.reset(random_obstacles=True)

    N = sim.num_robots
    collected = 0
    step = 0

    while collected < n_snapshots:
        # Prepare state
        robot_state, _ = policy.prepare_state(
            poses, distance, cos_val, sin_val, collision, a, goal_positions,
        )
        robot_obs = np.array(robot_state)

        robot_t = torch.tensor(robot_obs, dtype=torch.float32, device=device)
        obs_t = torch.tensor(obstacle_states, dtype=torch.float32, device=device)

        # ---- Precompute intermediates ----
        intermediates = precompute_attention_intermediates(
            attn_module, robot_t, obs_t, device,
        )

        # ---- Single-neighbor removal deltas ----
        single_records = compute_single_removal_deltas(
            intermediates, attn_module, policy_head, device,
        )
        for rec in single_records:
            for k in s_keys:
                s_scalar[k].append(rec[k])
            s_delta_attn.append(rec["delta_attn_out"])
            s_delta_emb.append(rec["delta_embedding"])
            s_delta_act.append(rec["delta_action"])
            s_snapshot.append(collected)

        # ---- Pair-removal deltas ----
        pair_records = compute_pair_removal_deltas(
            intermediates, attn_module, device, max_pairs=max_pairs,
        )
        for rec in pair_records:
            for k in p_keys:
                p_scalar[k].append(rec[k])
            p_snapshot.append(collected)

        # ---- Attention weight extraction (experiment 4) ----
        edge_index = intermediates["edge_index"]
        attn_w = intermediates["attn_weights"]
        robot_pos = intermediates["robot_position"][0]
        robot_head = intermediates["robot_heading"][0]
        obs_pos = intermediates["obs_position"][0]
        obs_head = intermediates["obs_heading"][0]
        n_obs = intermediates["n_obs"]

        for e in range(edge_index.shape[1]):
            tgt = edge_index[1, e].item()
            src = edge_index[0, e].item()
            w = attn_w[e].item()

            pos_i = robot_pos[tgt]
            heading_i = torch.atan2(robot_head[tgt, 1], robot_head[tgt, 0])

            if src < N:
                p_j = robot_pos[src]
                h_j = robot_head[src]
                is_robot_flag = 1
                src_id = src
            else:
                oi = src - N
                p_j = obs_pos[oi]
                h_j = obs_head[oi]
                is_robot_flag = 0
                src_id = oi

            rel_vec = p_j - pos_i
            d = torch.linalg.norm(rel_vec).item()
            ang = torch.atan2(rel_vec[1], rel_vec[0]).item()
            rel_ang = ang - heading_i.item()
            rel_ang = (rel_ang + np.pi) % (2 * np.pi) - np.pi

            hj_ang = torch.atan2(h_j[1], h_j[0]).item()
            rel_head = hj_ang - heading_i.item()
            rel_head = (rel_head + np.pi) % (2 * np.pi) - np.pi

            a_scalar["robot_i"].append(tgt)
            a_scalar["source_idx"].append(src_id)
            a_scalar["is_robot"].append(is_robot_flag)
            a_scalar["distance"].append(d)
            a_scalar["relative_angle"].append(rel_ang)
            a_scalar["rel_heading"].append(rel_head)
            a_scalar["attn_weight"].append(w)
            a_snapshot.append(collected)

        # ---- Step simulation ----
        raw_action, combined_weights = policy.get_action(
            robot_obs, obstacle_states, add_noise=False,
        )
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

    # ---- Pack into arrays ----
    single_data = {k: np.array(v) for k, v in s_scalar.items()}
    single_data["delta_attn_out"] = np.array(s_delta_attn)
    single_data["delta_embedding"] = np.array(s_delta_emb)
    single_data["delta_action"] = np.array(s_delta_act)
    single_data["snapshot_idx"] = np.array(s_snapshot)

    pair_data = {k: np.array(v) for k, v in p_scalar.items()}
    pair_data["snapshot_idx"] = np.array(p_snapshot)

    attn_data = {k: np.array(v) for k, v in a_scalar.items()}
    attn_data["snapshot_idx"] = np.array(a_snapshot)

    return single_data, pair_data, attn_data


# =====================================================================
# Section 3 — Experiment 1: Single-Neighbor Removal Sensitivity
# =====================================================================

def _run_regression_probe(X, y, cv=5):
    """Cross-validated Ridge regression R²."""
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())


def _run_mlp_probe(X, y, cv=5):
    """Cross-validated MLP regression R²."""
    pipe = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        ),
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())


def _run_rf_probe(X, y, cv=5):
    """Cross-validated RandomForest regression R²."""
    pipe = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())


def _run_classification_probe(X, y, cv=5):
    """Cross-validated Logistic Regression accuracy."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def experiment_1_sensitivity(
    single_data: Dict[str, np.ndarray],
    save_dir: Path,
    cv: int = 5,
    dpi: int = 150,
):
    """
    Experiment 1: Can you predict the geometric relationship from the
    delta vector alone?

    Trains linear (Ridge), nonlinear (MLP, RandomForest) probes on the
    delta vectors to predict distance, angle, velocity, and entity type.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — Single-Neighbor Removal Sensitivity")
    print("=" * 70)

    n_samples = single_data["distance"].shape[0]
    print(f"  Total perturbation records: {n_samples}")

    # ---- Input: delta vectors ----
    delta_emb = single_data["delta_embedding"]     # (S, 512)
    delta_attn = single_data["delta_attn_out"]     # (S, 256)

    # ---- Regression targets ----
    regression_targets = {
        "distance": single_data["distance"],
        "relative_angle": single_data["relative_angle"],
        "rel_heading_cos": single_data["rel_heading_cos"],
        "rel_heading_sin": single_data["rel_heading_sin"],
        "rel_velocity_lin": single_data["rel_velocity_lin"],
        "attn_weight": single_data["attn_weight"],
    }

    # ---- Classification target ----
    class_targets = {
        "entity_type": single_data["neighbor_type"],
    }

    # ---- Run probes ----
    probe_methods = {
        "Ridge": _run_regression_probe,
        "MLP": _run_mlp_probe,
        "RF": _run_rf_probe,
    }

    input_spaces = {
        "Δ_embedding": delta_emb,
        "Δ_attn_out": delta_attn,
    }

    results = {}
    for space_name, X in input_spaces.items():
        results[space_name] = {}
        print(f"\n  Input space: {space_name} (dim={X.shape[1]})")

        for tgt_name, y in regression_targets.items():
            for probe_name, probe_fn in probe_methods.items():
                key = f"{probe_name}_{tgt_name}"
                mean, std = probe_fn(X, y, cv)
                results[space_name][key] = (mean, std)
                print(f"    {probe_name:>6} → {tgt_name:<20}: R²={mean:.4f}±{std:.3f}")

        for tgt_name, y in class_targets.items():
            if len(np.unique(y)) < 2:
                continue
            mean, std = _run_classification_probe(X, y, cv)
            key = f"LogReg_{tgt_name}"
            results[space_name][key] = (mean, std)
            print(f"    LogReg → {tgt_name:<20}: Acc={mean:.4f}±{std:.3f}")

    # ---- Plot 1: Probe R² heatmap ----
    _plot_probe_heatmap(results, save_dir / "exp1_probe_heatmap.png", dpi)

    # ---- Plot 2: ||Δe|| vs distance (colored by entity type) ----
    delta_norms = np.linalg.norm(delta_emb, axis=1)
    distances = single_data["distance"]
    types = single_data["neighbor_type"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: delta norm vs distance
    ax = axes[0]
    robot_mask = types == 1
    obs_mask = types == 0
    ax.scatter(
        distances[robot_mask], delta_norms[robot_mask],
        alpha=0.3, s=8, c="tab:blue", label="Robot",
    )
    ax.scatter(
        distances[obs_mask], delta_norms[obs_mask],
        alpha=0.3, s=8, c="tab:red", label="Obstacle",
    )
    ax.set_xlabel("Distance to removed neighbor")
    ax.set_ylabel("‖Δe‖ (embedding delta norm)")
    ax.set_title("Embedding sensitivity vs distance")
    ax.legend(fontsize=9)

    # Scatter: delta norm vs relative angle
    ax = axes[1]
    ax.scatter(
        single_data["relative_angle"], delta_norms,
        alpha=0.2, s=6, c=distances, cmap="viridis",
    )
    cb = fig.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cb.set_label("Distance")
    ax.set_xlabel("Relative angle to removed neighbor (rad)")
    ax.set_ylabel("‖Δe‖ (embedding delta norm)")
    ax.set_title("Embedding sensitivity vs angle")

    fig.suptitle("Experiment 1: Single-Neighbor Removal Sensitivity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "exp1_delta_vs_geometry.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✅ Saved exp1_delta_vs_geometry.png")

    # ---- Plot 3: delta norm vs attention weight ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        single_data["attn_weight"], delta_norms,
        alpha=0.2, s=6, c=distances, cmap="viridis",
    )
    cb = fig.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cb.set_label("Distance")
    ax.set_xlabel("Baseline attention weight")
    ax.set_ylabel("‖Δe‖ (embedding delta norm)")
    ax.set_title("Embedding delta norm vs attention weight")
    fig.tight_layout()
    fig.savefig(save_dir / "exp1_delta_vs_attn.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved exp1_delta_vs_attn.png")

    # ---- Interpretation ----
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    emb_dist_ridge = results["Δ_embedding"].get("Ridge_distance", (0, 0))[0]
    emb_dist_mlp = results["Δ_embedding"].get("MLP_distance", (0, 0))[0]
    emb_angle_ridge = results["Δ_embedding"].get("Ridge_relative_angle", (0, 0))[0]
    emb_angle_mlp = results["Δ_embedding"].get("MLP_relative_angle", (0, 0))[0]
    emb_type_acc = results["Δ_embedding"].get("LogReg_entity_type", (0, 0))[0]

    print(f"  Distance from Δe:  Ridge R²={emb_dist_ridge:.3f}  MLP R²={emb_dist_mlp:.3f}")
    if emb_dist_mlp > 0.3:
        print("    → ✅ Distance is decodable from Δe — the embedding encodes")
        print("      directional spatial information about each neighbor.")
    elif emb_dist_ridge > 0.1:
        print("    → Partial: distance is weakly encoded (linear R²>0.1).")
    else:
        print("    → ❌ Distance is NOT decodable from Δe.")

    print(f"  Angle from Δe:     Ridge R²={emb_angle_ridge:.3f}  MLP R²={emb_angle_mlp:.3f}")
    if emb_angle_mlp > 0.3:
        print("    → ✅ Relative angle is decodable — directional encoding confirmed.")

    print(f"  Entity type (Acc): {emb_type_acc:.3f}")
    if emb_type_acc > 0.7:
        print("    → ✅ Robot vs obstacle produces systematically different Δe.")
    print("-" * 70)

    return results


# =====================================================================
# Section 4 — Experiment 2: Superposition Test
# =====================================================================

def experiment_2_superposition(
    pair_data: Dict[str, np.ndarray],
    save_dir: Path,
    dpi: int = 150,
):
    """
    Experiment 2: Check whether Δe^{jk} ≈ Δe^j + Δe^k.

    If additive, the embedding encodes neighbors independently.
    If not, the attention mechanism creates interaction effects.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — Superposition Test")
    print("=" * 70)

    n_pairs = pair_data["delta_jk_norm"].shape[0]
    print(f"  Total pair records: {n_pairs}")

    delta_jk = pair_data["delta_jk_norm"]
    delta_sum = pair_data["delta_sum_norm"]
    residual = pair_data["residual_norm"]
    rel_resid = pair_data["relative_residual"]
    cos_sim = pair_data["cosine_sim"]
    dist_jk = pair_data["dist_jk"]
    dist_ij = pair_data["dist_ij"]
    dist_ik = pair_data["dist_ik"]

    # ---- Summary statistics ----
    mean_rel_resid = np.mean(rel_resid)
    median_rel_resid = np.median(rel_resid)
    mean_cos_sim = np.mean(cos_sim)
    median_cos_sim = np.median(cos_sim)

    print(f"\n  Relative residual ‖Δ_jk − (Δ_j+Δ_k)‖ / ‖Δ_jk‖:")
    print(f"    Mean   = {mean_rel_resid:.4f}")
    print(f"    Median = {median_rel_resid:.4f}")
    print(f"  Cosine similarity cos(Δ_jk, Δ_j+Δ_k):")
    print(f"    Mean   = {mean_cos_sim:.4f}")
    print(f"    Median = {median_cos_sim:.4f}")

    # ---- Plot 1: ||Δe_jk|| vs ||Δe_j + Δe_k|| (diagonal = perfect additivity) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0, 0]
    ax.scatter(delta_sum, delta_jk, alpha=0.15, s=6, c="tab:blue")
    lim = max(delta_jk.max(), delta_sum.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect additivity")
    ax.set_xlabel("‖Δe_j + Δe_k‖")
    ax.set_ylabel("‖Δe_jk‖ (actual pair removal)")
    ax.set_title("Superposition: actual vs sum of singles")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    # ---- Plot 2: Histogram of relative residual ----
    ax = axes[0, 1]
    ax.hist(rel_resid, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(
        mean_rel_resid, color="red", ls="--", lw=1.5,
        label=f"Mean={mean_rel_resid:.3f}",
    )
    ax.set_xlabel("Relative residual ‖Δ_jk − (Δ_j+Δ_k)‖ / ‖Δ_jk‖")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of superposition residual")
    ax.legend(fontsize=9)

    # ---- Plot 3: Residual vs dist_jk (interaction vs proximity) ----
    ax = axes[1, 0]
    ax.scatter(dist_jk, rel_resid, alpha=0.15, s=6, c="tab:green")
    ax.set_xlabel("Distance between neighbors j and k")
    ax.set_ylabel("Relative residual")
    ax.set_title("Interaction strength vs neighbor proximity")

    # ---- Plot 4: Cosine similarity distribution ----
    ax = axes[1, 1]
    ax.hist(cos_sim, bins=60, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(
        mean_cos_sim, color="red", ls="--", lw=1.5,
        label=f"Mean={mean_cos_sim:.3f}",
    )
    ax.set_xlabel("Cosine similarity cos(Δ_jk, Δ_j+Δ_k)")
    ax.set_ylabel("Count")
    ax.set_title("Directional alignment of pair delta with sum")
    ax.legend(fontsize=9)

    fig.suptitle("Experiment 2: Superposition Test", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "exp2_superposition.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✅ Saved exp2_superposition.png")

    # ---- Interpretation ----
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if mean_rel_resid < 0.15 and mean_cos_sim > 0.9:
        print("  ✅ Approximately ADDITIVE: Δe_jk ≈ Δe_j + Δe_k.")
        print("     Each neighbor's contribution is roughly independent.")
        print("     → Group composition can be reasoned about by summing")
        print("       individual contributions.")
    elif mean_rel_resid < 0.3:
        print("  ⚠️  Weakly additive (residual < 30%).")
        print("     Some interaction effects exist but are moderate.")
    else:
        print("  ❌ NON-ADDITIVE: significant interaction effects.")
        print("     The attention mechanism creates context-dependent encoding")
        print("     where one neighbor changes how another is represented.")
        print("     → Need full graph context for accurate group embeddings.")

    # Check if nearby neighbors interact more
    close_mask = dist_jk < np.median(dist_jk)
    far_mask = ~close_mask
    if close_mask.sum() > 10 and far_mask.sum() > 10:
        close_resid = rel_resid[close_mask].mean()
        far_resid = rel_resid[far_mask].mean()
        print(f"\n  Proximity effect on interaction:")
        print(f"    Close pairs (d_jk < median): mean residual = {close_resid:.4f}")
        print(f"    Far pairs   (d_jk > median): mean residual = {far_resid:.4f}")
        if close_resid > far_resid * 1.2:
            print("    → Close neighbors interact MORE (non-additive).")
        else:
            print("    → No strong proximity effect on interaction strength.")
    print("-" * 70)


# =====================================================================
# Section 5 — Experiment 3: Behavioral Relevance of Deltas
# =====================================================================

def experiment_3_behavioral_relevance(
    single_data: Dict[str, np.ndarray],
    save_dir: Path,
    cv: int = 5,
    dpi: int = 150,
):
    """
    Experiment 3: Does removing neighbor j change the action output?

    Correlate ||Δa_i|| with geometric features and ||Δe_i||.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — Behavioral Relevance of Deltas")
    print("=" * 70)

    delta_action = single_data["delta_action"]       # (S, 2)
    delta_emb = single_data["delta_embedding"]       # (S, 512)
    distances = single_data["distance"]
    types = single_data["neighbor_type"]
    attn_w = single_data["attn_weight"]

    action_norms = np.linalg.norm(delta_action, axis=1)
    emb_norms = np.linalg.norm(delta_emb, axis=1)
    lin_delta = np.abs(delta_action[:, 0])
    ang_delta = np.abs(delta_action[:, 1])

    n = len(distances)
    print(f"  Total perturbation records: {n}")

    # ---- Correlation analysis ----
    from scipy.stats import pearsonr, spearmanr

    features = {
        "distance": distances,
        "attn_weight": attn_w,
        "‖Δe‖": emb_norms,
    }
    print("\n  Correlations with ‖Δa‖:")
    for name, feat in features.items():
        r_p, p_p = pearsonr(feat, action_norms)
        r_s, p_s = spearmanr(feat, action_norms)
        print(f"    {name:<15}: Pearson r={r_p:+.4f} (p={p_p:.2e})  "
              f"Spearman ρ={r_s:+.4f} (p={p_s:.2e})")

    # ---- Probe: predict Δa from Δe ----
    print("\n  Probes: predict Δa components from Δe (512-dim):")
    for tgt_name, y in [("Δ_lin_vel", delta_action[:, 0]),
                         ("Δ_ang_vel", delta_action[:, 1])]:
        r2_ridge, std_r = _run_regression_probe(delta_emb, y, cv)
        r2_mlp, std_m = _run_mlp_probe(delta_emb, y, cv)
        print(f"    {tgt_name:<12}: Ridge R²={r2_ridge:.4f}±{std_r:.3f}  "
              f"MLP R²={r2_mlp:.4f}±{std_m:.3f}")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (a) ||Δa|| vs distance
    ax = axes[0, 0]
    robot_mask = types == 1
    obs_mask = types == 0
    ax.scatter(
        distances[robot_mask], action_norms[robot_mask],
        alpha=0.2, s=6, c="tab:blue", label="Robot",
    )
    ax.scatter(
        distances[obs_mask], action_norms[obs_mask],
        alpha=0.2, s=6, c="tab:red", label="Obstacle",
    )
    ax.set_xlabel("Distance to removed neighbor")
    ax.set_ylabel("‖Δa‖ (action delta norm)")
    ax.set_title("Action sensitivity vs distance")
    ax.legend(fontsize=9)

    # (b) ||Δa|| vs ||Δe||
    ax = axes[0, 1]
    ax.scatter(emb_norms, action_norms, alpha=0.15, s=6, c=distances, cmap="viridis")
    cb = fig.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cb.set_label("Distance")
    ax.set_xlabel("‖Δe‖ (embedding delta norm)")
    ax.set_ylabel("‖Δa‖ (action delta norm)")
    ax.set_title("Action vs embedding sensitivity")

    # (c) |Δ_lin_vel| vs distance
    ax = axes[1, 0]
    ax.scatter(distances, lin_delta, alpha=0.15, s=6, c="tab:green")
    ax.set_xlabel("Distance to removed neighbor")
    ax.set_ylabel("|Δ linear velocity|")
    ax.set_title("Linear velocity sensitivity")

    # (d) |Δ_ang_vel| vs distance
    ax = axes[1, 1]
    ax.scatter(distances, ang_delta, alpha=0.15, s=6, c="tab:orange")
    ax.set_xlabel("Distance to removed neighbor")
    ax.set_ylabel("|Δ angular velocity|")
    ax.set_title("Angular velocity sensitivity")

    fig.suptitle(
        "Experiment 3: Behavioral Relevance of Deltas",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        save_dir / "exp3_behavioral_relevance.png", dpi=dpi, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"\n  ✅ Saved exp3_behavioral_relevance.png")

    # ---- Interpretation ----
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    r_dist, _ = pearsonr(distances, action_norms)
    r_emb, _ = pearsonr(emb_norms, action_norms)

    if r_dist < -0.3:
        print("  ✅ Removing CLOSE neighbors changes action more (r_dist < -0.3).")
        print("     → Policy encodes behaviorally relevant proximity.")
    elif r_dist < -0.1:
        print("  ⚠️  Weak negative correlation between distance and ||Δa||.")
    else:
        print("  ❌ No strong distance-dependence in action sensitivity.")

    if r_emb > 0.5:
        print(f"  ✅ Tight coupling: ‖Δe‖ and ‖Δa‖ are correlated (r={r_emb:.3f}).")
        print("     → Embedding perturbation maps consistently to action perturbation.")
        print("     → Limited slack for coordination layer to exploit.")
    elif r_emb > 0.2:
        print(f"  ⚠️  Moderate coupling (r={r_emb:.3f}). Some slack exists.")
    else:
        print(f"  ❌ Weak coupling (r={r_emb:.3f}). Embedding and action deltas")
        print("     are loosely connected — coordination layer can exploit this.")

    # Compare robot vs obstacle sensitivity
    if robot_mask.sum() > 10 and obs_mask.sum() > 10:
        mean_robot = action_norms[robot_mask].mean()
        mean_obs = action_norms[obs_mask].mean()
        print(f"\n  Entity type effect on ‖Δa‖:")
        print(f"    Robot neighbors:    mean ‖Δa‖ = {mean_robot:.4f}")
        print(f"    Obstacle neighbors: mean ‖Δa‖ = {mean_obs:.4f}")
    print("-" * 70)


# =====================================================================
# Section 6 — Experiment 4: Attention Weight Inspection
# =====================================================================

def experiment_4_attention_weights(
    attn_data: Dict[str, np.ndarray],
    save_dir: Path,
    dpi: int = 150,
):
    """
    Experiment 4: Extract attention weights and plot against distance,
    heading, and entity type.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 — Attention Weight Inspection")
    print("=" * 70)

    distances = attn_data["distance"]
    rel_angles = attn_data["relative_angle"]
    rel_headings = attn_data["rel_heading"]
    weights = attn_data["attn_weight"]
    is_robot = attn_data["is_robot"]
    n_records = len(distances)

    print(f"  Total edge records: {n_records}")
    robot_mask = is_robot == 1
    obs_mask = is_robot == 0
    print(f"  Robot edges: {robot_mask.sum()},  Obstacle edges: {obs_mask.sum()}")

    # ---- Summary statistics ----
    if robot_mask.sum() > 0:
        print(f"\n  Robot neighbors:")
        print(f"    Mean weight = {weights[robot_mask].mean():.4f}")
        print(f"    Mean dist   = {distances[robot_mask].mean():.3f}")
    if obs_mask.sum() > 0:
        print(f"  Obstacle neighbors:")
        print(f"    Mean weight = {weights[obs_mask].mean():.4f}")
        print(f"    Mean dist   = {distances[obs_mask].mean():.3f}")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Attention weight vs distance
    ax = axes[0, 0]
    if robot_mask.sum() > 0:
        ax.scatter(
            distances[robot_mask], weights[robot_mask],
            alpha=0.15, s=6, c="tab:blue", label="Robot",
        )
    if obs_mask.sum() > 0:
        ax.scatter(
            distances[obs_mask], weights[obs_mask],
            alpha=0.15, s=6, c="tab:red", label="Obstacle",
        )
    ax.set_xlabel("Distance to neighbor")
    ax.set_ylabel("Attention weight")
    ax.set_title("Attention weight vs distance")
    ax.legend(fontsize=9)

    # (b) Attention weight vs relative angle
    ax = axes[0, 1]
    sc = ax.scatter(
        rel_angles, weights, alpha=0.15, s=6, c=distances, cmap="viridis",
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Distance")
    ax.set_xlabel("Relative angle (rad)")
    ax.set_ylabel("Attention weight")
    ax.set_title("Attention weight vs bearing angle")

    # (c) 2D binned heatmap: distance × angle → mean attention weight
    ax = axes[1, 0]
    n_dist_bins = 15
    n_angle_bins = 12
    d_bins = np.linspace(distances.min(), distances.max() + 1e-6, n_dist_bins + 1)
    a_bins = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    heatmap = np.full((n_dist_bins, n_angle_bins), np.nan)
    for di in range(n_dist_bins):
        for ai in range(n_angle_bins):
            mask = (
                (distances >= d_bins[di]) & (distances < d_bins[di + 1])
                & (rel_angles >= a_bins[ai]) & (rel_angles < a_bins[ai + 1])
            )
            if mask.sum() > 2:
                heatmap[di, ai] = weights[mask].mean()

    im = ax.imshow(
        heatmap, origin="lower", aspect="auto", cmap="YlOrRd",
        extent=[a_bins[0], a_bins[-1], d_bins[0], d_bins[-1]],
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean attn weight")
    ax.set_xlabel("Relative angle (rad)")
    ax.set_ylabel("Distance")
    ax.set_title("Attention weight heatmap (distance × angle)")

    # (d) Box plot by entity type
    ax = axes[1, 1]
    data_box = []
    labels_box = []
    if robot_mask.sum() > 0:
        data_box.append(weights[robot_mask])
        labels_box.append("Robot")
    if obs_mask.sum() > 0:
        data_box.append(weights[obs_mask])
        labels_box.append("Obstacle")
    if data_box:
        bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True)
        colors = ["tab:blue", "tab:red"]
        for patch, color in zip(bp["boxes"], colors[: len(data_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Attention weight")
    ax.set_title("Attention weight by entity type")

    fig.suptitle(
        "Experiment 4: Attention Weight Inspection",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        save_dir / "exp4_attention_weights.png", dpi=dpi, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"\n  ✅ Saved exp4_attention_weights.png")

    # ---- Correlation analysis ----
    from scipy.stats import pearsonr, spearmanr

    print("\n  Correlations with attention weight:")
    for name, feat in [("distance", distances), ("relative_angle", np.abs(rel_angles)),
                        ("abs_rel_heading", np.abs(rel_headings))]:
        r, p = spearmanr(feat, weights)
        print(f"    {name:<20}: Spearman ρ={r:+.4f} (p={p:.2e})")

    # ---- Distance-binned analysis ----
    dist_quantiles = np.quantile(distances, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\n  Distance-binned mean attention weight:")
    for lo, hi in zip(dist_quantiles[:-1], dist_quantiles[1:]):
        m = (distances >= lo) & (distances < hi + 1e-6)
        if m.sum() > 0:
            print(f"    [{lo:.2f}, {hi:.2f}): n={m.sum():>5}  "
                  f"mean_w={weights[m].mean():.4f}")

    # ---- Interpretation ----
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    r_dist, _ = spearmanr(distances, weights)
    if r_dist < -0.2:
        print("  ✅ Closer neighbors get HIGHER attention weight.")
        print("     → Distance-based relational encoding is present.")
    elif abs(r_dist) < 0.1:
        print("  ❌ No clear distance dependence in attention weights.")
    else:
        print(f"  ⚠️  Unexpected positive distance-weight correlation (ρ={r_dist:.3f}).")

    if robot_mask.sum() > 10 and obs_mask.sum() > 10:
        mean_w_r = weights[robot_mask].mean()
        mean_w_o = weights[obs_mask].mean()
        if abs(mean_w_r - mean_w_o) / max(mean_w_r, mean_w_o, 1e-8) > 0.15:
            print(f"  ✅ Different attention patterns for robots ({mean_w_r:.4f})"
                  f" vs obstacles ({mean_w_o:.4f}).")
        else:
            print("  Entity type does not strongly affect attention weight.")
    print("-" * 70)


# =====================================================================
# Section 7 — Plotting Helpers
# =====================================================================

def _plot_probe_heatmap(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    save_path: Path,
    dpi: int = 150,
):
    """Plot a heatmap of probe R² / accuracy across input spaces and targets."""
    row_labels = list(results.keys())
    all_targets = []
    for targets in results.values():
        for t in targets:
            if t not in all_targets:
                all_targets.append(t)
    col_labels = all_targets

    matrix = np.zeros((len(row_labels), len(col_labels)))
    annot = [["" for _ in col_labels] for _ in row_labels]

    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            if col in results[row]:
                mean, std = results[row][col]
                matrix[i, j] = mean
                annot[i][j] = f"{mean:.3f}\n±{std:.3f}"
            else:
                matrix[i, j] = np.nan
                annot[i][j] = "N/A"

    fig, ax = plt.subplots(
        figsize=(max(12, len(col_labels) * 1.5), max(3, len(row_labels) * 1.0)),
    )
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="R² / Accuracy")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            color = "black" if matrix[i, j] > 0.5 else "white"
            ax.text(
                j, i, annot[i][j], ha="center", va="center",
                fontsize=6, color=color,
            )

    ax.set_title(
        "Experiment 1: Probe R² / Accuracy from Δe → geometric features",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path.name}")


# =====================================================================
# Section 8 — Main
# =====================================================================

def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 70)
    print("GAT Perturbation Experiments")
    print("  1. Single-neighbor removal sensitivity")
    print("  2. Superposition test")
    print("  3. Behavioral relevance of deltas")
    print("  4. Attention weight inspection")
    print("=" * 70)

    # ---- Load model and environment ----
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

    # ---- Collect perturbation data ----
    print(f"\n[2/6] Collecting {cfg['n_snapshots']} snapshots with perturbation analysis ...")
    single_data, pair_data, attn_data = collect_perturbation_data(
        policy, sim, device,
        n_snapshots=cfg["n_snapshots"],
        max_steps=cfg["max_steps_per_episode"],
        max_pairs=cfg["max_pairs_per_robot"],
    )
    print(f"  Single-removal records:  {single_data['distance'].shape[0]}")
    print(f"  Pair-removal records:    {pair_data['delta_jk_norm'].shape[0]}")
    print(f"  Attention edge records:  {attn_data['distance'].shape[0]}")

    # Save raw data
    np.savez_compressed(save_dir / "single_data.npz", **single_data)
    np.savez_compressed(save_dir / "pair_data.npz", **pair_data)
    np.savez_compressed(save_dir / "attn_data.npz", **attn_data)
    print(f"  Saved raw data to {save_dir}")

    # ---- Experiment 1 ----
    print("\n[3/6] Running Experiment 1 ...")
    experiment_1_sensitivity(
        single_data, save_dir,
        cv=cfg["cv_folds"], dpi=cfg["dpi"],
    )

    # ---- Experiment 2 ----
    print("\n[4/6] Running Experiment 2 ...")
    experiment_2_superposition(
        pair_data, save_dir, dpi=cfg["dpi"],
    )

    # ---- Experiment 3 ----
    print("\n[5/6] Running Experiment 3 ...")
    experiment_3_behavioral_relevance(
        single_data, save_dir,
        cv=cfg["cv_folds"], dpi=cfg["dpi"],
    )

    # ---- Experiment 4 ----
    print("\n[6/6] Running Experiment 4 ...")
    experiment_4_attention_weights(
        attn_data, save_dir, dpi=cfg["dpi"],
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"  Figures and data saved to: {save_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
