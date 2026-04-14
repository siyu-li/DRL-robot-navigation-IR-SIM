"""
Collect and save GAT embedding data for offline analysis.

Rolls out the frozen TD3 policy through the simulation and records, for every
snapshot (time-step):

  Per-robot scalars / arrays (shape S × N × ...):
    position          (S, N, 3)   — (x, y, θ) in world frame
    goal_position     (S, N, 2)   — (gx, gy) in world frame
    dist_to_goal      (S, N)      — Euclidean distance to goal
    heading_error     (S, N)      — angular error ∈ [0, π]
    self_embedding    (S, N, 256) — robot's own encoded features, pre-decoder
    attn_embedding    (S, N, 256) — aggregated neighbor messages, pre-decoder
    pre_decode        (S, N, 512) — [self ∥ attn], pre-decoder
    full_embedding    (S, N, 512) — after decoder

  Per-snapshot pairwise (shape S × N × N or S × N × M):
    dist_rr           (S, N, N)       — robot-robot centre-to-centre distances (m)
    dist_ro           (S, N, N_obs)   — robot-obstacle centre-to-centre distances (m)
    hard_weights_rr   (S, N, N)       — hard attention mask (robot-robot)

All arrays are saved as a single compressed .npz file together with small
metadata scalars (num_robots, num_obstacles, embedding_dim).

Usage:
    python -m robot_nav.scripts.analysis.collect_gat_data
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
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

    # Collection
    "n_snapshots": 1000,
    "max_steps_per_episode": 200,

    # Output
    "save_path": "robot_nav/analysis/collected_data/gat_data.npz",

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =====================================================================
# Collection
# =====================================================================
def collect(
    policy: TD3Obstacle,
    sim: MARL_SIM_OBSTACLE,
    device: torch.device,
    n_snapshots: int,
    max_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Roll out the frozen policy and collect per-snapshot data.

    Returns a dict whose arrays all have a leading S (snapshot) dimension.
    Per-robot arrays have shape (S, N, ...).
    Pairwise arrays have shape (S, N, N) or (S, N, N_obs).
    """
    N = sim.num_robots
    M = sim.num_obstacles

    SIGMA_2 = 2.0   # narrow bandwidth (m)
    SIGMA_4 = 4.0   # wide bandwidth (m)

    # Per-snapshot accumulators
    pos_list:        List[np.ndarray] = []   # (N, 3)
    goal_list:       List[np.ndarray] = []   # (N, 2)
    dtg_list:        List[np.ndarray] = []   # (N,)
    herr_list:       List[np.ndarray] = []   # (N,)
    self_emb_list:   List[np.ndarray] = []   # (N, 256)
    attn_emb_list:   List[np.ndarray] = []   # (N, 256)
    pre_dec_list:    List[np.ndarray] = []   # (N, 512)
    full_emb_list:   List[np.ndarray] = []   # (N, 512)
    dist_rr_list:    List[np.ndarray] = []   # (N, N)
    dist_ro_list:    List[np.ndarray] = []   # (N, M)
    hard_rr_list:    List[np.ndarray] = []   # (N, N)
    density_s2_list: List[np.ndarray] = []   # (N,)  σ=2m
    density_s4_list: List[np.ndarray] = []   # (N,)  σ=4m

    (
        poses, distance, cos_val, sin_val, collision, goal, a, reward,
        positions, goal_positions, obstacle_states,
    ) = sim.reset(random_obstacles=True)

    collected = 0
    step = 0

    while collected < n_snapshots:
        # ---- Prepare state ----
        robot_state, _ = policy.prepare_state(
            poses, distance, cos_val, sin_val, collision, a, goal_positions
        )
        robot_obs = np.array(robot_state)  # (N, state_dim)

        robot_t = torch.tensor(robot_obs, dtype=torch.float32, device=device).unsqueeze(0)      # (1, N, 11)
        obs_t   = torch.tensor(obstacle_states, dtype=torch.float32, device=device).unsqueeze(0) # (1, M, 4)

        # ---- Single forward pass — get everything at once ----
        with torch.no_grad():
            (
                att_embedding,   # (N, 512)  — B=1 so B*N = N
                _,               # hard_logits_rr  (not needed)
                _,               # hard_logits_ro  (not needed)
                unnorm_dist_rr,  # (N, N, 1)
                unnorm_dist_ro,  # (N, M, 1)
                _,               # mean_entropy    (not needed)
                hard_w_rr,       # (1, N, N)
                _,               # hard_w_ro       (not needed)
                _,               # combined_w      (not needed)
            ) = policy.actor.attention(robot_t, obs_t)

        # _pre_decoder_embedding is stored on the module by forward()
        pre_dec = policy.actor.attention._pre_decoder_embedding.cpu().numpy()  # (N, 512)
        embed_dim = pre_dec.shape[1] // 2  # 256

        # ---- Decompose pre_decode into self and attn halves ----
        self_emb = pre_dec[:, :embed_dim]   # (N, 256)
        attn_emb = pre_dec[:, embed_dim:]   # (N, 256)
        full_emb = att_embedding.cpu().numpy()   # (N, 512)

        # ---- Pairwise distances — already in metres ----
        drr = unnorm_dist_rr.squeeze(-1).cpu().numpy()  # (N, N)
        dro = unnorm_dist_ro.squeeze(-1).cpu().numpy()  # (N, M)
        hrr = hard_w_rr.squeeze(0).cpu().numpy()        # (N, N)

        # ---- Per-robot scalars ----
        pos_snap  = np.array([[px, py, th] for px, py, th in poses])  # (N, 3)
        goal_snap = np.array([[gx, gy] for gx, gy in goal_positions]) # (N, 2)

        dtg  = np.array([
            np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            for (px, py, _), (gx, gy) in zip(poses, goal_positions)
        ])  # (N,)

        herr = np.array([
            min(abs(d := np.arctan2(gy - py, gx - px) - th),
                2 * np.pi - abs(d))
            for (px, py, th), (gx, gy) in zip(poses, goal_positions)
        ])  # (N,)

        # ---- Gaussian density scores per robot ----
        # drr: (N, N)  centre-to-centre robot-robot distances
        # dro: (N, M)  centre-to-centre robot-obstacle distances
        # Exclude self (diagonal) from robot-robot sum via the eye mask.
        eye_mask = np.eye(N, dtype=bool)  # (N, N)

        def _gaussian_density(sigma: float) -> np.ndarray:
            # robot-robot contribution (exclude i==j)
            rr = np.exp(-drr ** 2 / (2 * sigma ** 2))
            rr[eye_mask] = 0.0
            rr_sum = rr.sum(axis=1)          # (N,)
            # robot-obstacle contribution
            ro_sum = np.exp(-dro ** 2 / (2 * sigma ** 2)).sum(axis=1)  # (N,)
            return rr_sum + ro_sum

        density_s2 = _gaussian_density(SIGMA_2)  # (N,)
        density_s4 = _gaussian_density(SIGMA_4)  # (N,)

        # ---- Accumulate ----
        pos_list.append(pos_snap)
        goal_list.append(goal_snap)
        dtg_list.append(dtg)
        herr_list.append(herr)
        self_emb_list.append(self_emb)
        attn_emb_list.append(attn_emb)
        pre_dec_list.append(pre_dec)
        full_emb_list.append(full_emb)
        dist_rr_list.append(drr)
        dist_ro_list.append(dro)
        hard_rr_list.append(hrr)
        density_s2_list.append(density_s2)
        density_s4_list.append(density_s4)

        collected += 1
        if collected % 100 == 0:
            print(f"  {collected}/{n_snapshots} snapshots collected ...")

        # ---- Step the environment ----
        raw_action, combined_weights = policy.get_action(
            robot_obs, obstacle_states, add_noise=False
        )
        a_in = [[(act[0] + 1) / 4, act[1]] for act in raw_action]
        (
            poses, distance, cos_val, sin_val, collision, goal, a, reward,
            positions, goal_positions, obstacle_states,
        ) = sim.step(a_in, None, combined_weights)

        step += 1
        if any(collision) or step >= max_steps:
            (
                poses, distance, cos_val, sin_val, collision, goal, a, reward,
                positions, goal_positions, obstacle_states,
            ) = sim.reset(random_obstacles=True)
            step = 0

    return {
        # (S, N, 3) / (S, N, 2)
        "position":        np.stack(pos_list),
        "goal_position":   np.stack(goal_list),
        # (S, N)
        "dist_to_goal":    np.stack(dtg_list),
        "heading_error":   np.stack(herr_list),
        # (S, N, D)
        "self_embedding":  np.stack(self_emb_list),
        "attn_embedding":  np.stack(attn_emb_list),
        "pre_decode":      np.stack(pre_dec_list),
        "full_embedding":  np.stack(full_emb_list),
        # (S, N, N) / (S, N, M)
        "dist_rr":         np.stack(dist_rr_list),
        "dist_ro":         np.stack(dist_ro_list),
        "hard_weights_rr": np.stack(hard_rr_list),
        # (S, N)  Gaussian kernel density scores
        "density_sigma2":  np.stack(density_s2_list),
        "density_sigma4":  np.stack(density_s4_list),
    }


# =====================================================================
# Main
# =====================================================================
def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    save_path = Path(cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GAT Data Collection")
    print("=" * 60)

    # ---- Load environment ----
    print("\n[1/3] Loading environment and frozen policy ...")
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
    print(f"  Model:   {cfg['model_name']}")
    print(f"  Robots:  {sim.num_robots}  |  Obstacles: {sim.num_obstacles}")
    print(f"  Device:  {device}")

    # ---- Collect ----
    print(f"\n[2/3] Collecting {cfg['n_snapshots']} snapshots ...")
    data = collect(
        policy, sim, device,
        n_snapshots=cfg["n_snapshots"],
        max_steps=cfg["max_steps_per_episode"],
    )

    # ---- Print summary ----
    S, N = data["dist_to_goal"].shape
    print(f"\n  Snapshots : {S}")
    print(f"  Robots    : {N}")
    print(f"  Obstacles : {data['dist_ro'].shape[2]}")
    print(f"  Embedding dims — self: {data['self_embedding'].shape[2]}, "
          f"attn: {data['attn_embedding'].shape[2]}, "
          f"full: {data['full_embedding'].shape[2]}")
    print(f"\n  Array shapes:")
    for k, v in data.items():
        print(f"    {k:<20} {str(v.shape):<25} dtype={v.dtype}")

    # ---- Save ----
    print(f"\n[3/3] Saving to {save_path} ...")
    np.savez_compressed(
        save_path,
        **data,
        # Metadata scalars (retrieved via data['num_robots'].item())
        num_robots=np.array(N),
        num_obstacles=np.array(data["dist_ro"].shape[2]),
        embedding_dim=np.array(data["self_embedding"].shape[2]),
    )
    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"  Saved  ({size_mb:.1f} MB)")
    print("=" * 60)

    # ---- Distribution histograms ----
    print("\nPlotting distributions ...")

    ROBOT_RADIUS    = 0.2
    OBSTACLE_RADIUS = 0.7

    # dist_to_goal: goal is a point, no radius subtraction
    dtg_flat = data["dist_to_goal"].ravel()

    # Robot-robot proximity: subtract both robot radii, exclude self-distance diagonal
    drr     = data["dist_rr"]                            # (S, N, N)
    mask_rr = ~np.eye(N, dtype=bool)[np.newaxis, :, :]  # (1, N, N)
    prox_rr_flat = (drr - 2 * ROBOT_RADIUS)[np.broadcast_to(mask_rr, drr.shape)].ravel()

    # Robot-obstacle proximity: subtract one robot radius and one obstacle radius
    prox_ro_flat = (data["dist_ro"] - ROBOT_RADIUS - OBSTACLE_RADIUS).ravel()

    dens_s2_flat = data["density_sigma2"].ravel()
    dens_s4_flat = data["density_sigma4"].ravel()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Collected Data — Proximity & Density Distributions", fontsize=13, fontweight="bold")
    axes = axes.ravel()

    def _hist_with_percentiles(ax, arr, color, xlabel, title, vline_zero=False):
        ax.hist(arr, bins=60, color=color, edgecolor="white", linewidth=0.4)
        p33v, p67v = np.percentile(arr, 33), np.percentile(arr, 67)
        ax.axvline(p33v, color="orange", linestyle="--", linewidth=1.2, label=f"p33={p33v:.2f}")
        ax.axvline(p67v, color="red",    linestyle="--", linewidth=1.2, label=f"p67={p67v:.2f}")
        if vline_zero:
            ax.axvline(0, color="black", linestyle=":", linewidth=1.0, label="contact (0m)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Row 0 — proximity
    _hist_with_percentiles(axes[0], dtg_flat,    "steelblue", "Distance to goal (m)",    "dist_to_goal")
    _hist_with_percentiles(axes[1], prox_rr_flat,"seagreen",  "Robot-robot proximity (m)",
                           f"prox_rr  (c2c − {2*ROBOT_RADIUS:.1f}m)", vline_zero=True)
    _hist_with_percentiles(axes[2], prox_ro_flat,"salmon",    "Robot-obstacle proximity (m)",
                           f"prox_ro  (c2c − {ROBOT_RADIUS+OBSTACLE_RADIUS:.1f}m)", vline_zero=True)

    # Row 1 — density scores
    _hist_with_percentiles(axes[3], dens_s2_flat, "mediumpurple", "Density score",
                           "density  σ=2m  (per robot)")
    _hist_with_percentiles(axes[4], dens_s4_flat, "darkorchid",   "Density score",
                           "density  σ=4m  (per robot)")
    axes[5].axis("off")   # spare panel — left blank

    fig.tight_layout()
    hist_path = save_path.parent / "distance_distributions.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Histogram saved to {hist_path}")

    # ---- Print percentile summary ----
    print("\nPercentile summary (useful for threshold setting):")
    for name, arr in [("dist_to_goal", dtg_flat), ("prox_rr", prox_rr_flat), ("prox_ro", prox_ro_flat),
                      ("density_sigma2", dens_s2_flat), ("density_sigma4", dens_s4_flat)]:
        p10, p25, p33, p50, p67, p75, p90 = np.percentile(arr, [10, 25, 33, 50, 67, 75, 90])
        print(f"  {name:<20}  p10={p10:.2f}  p25={p25:.2f}  p33={p33:.2f}  "
              f"p50={p50:.2f}  p67={p67:.2f}  p75={p75:.2f}  p90={p90:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
