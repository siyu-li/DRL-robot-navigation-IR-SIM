"""
RL Training Script for PPO Group Switcher (14 Robots).

Trains a categorical PPO policy that selects which robot subgroup to
activate every ``selection_interval`` simulation steps.  The frozen
decentralized TD3Obstacle policy generates low-level actions; the
switcher only decides *which* group moves.

Usage:
    python -m robot_nav.scripts.train_switcher_rl

Edit the CONFIG dictionary below to change hyperparameters.
"""

import logging
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.groups.group_generator import (
    generate_all_groups,
    filter_groups_by_size,
)
from robot_nav.models.MARL.switcher.rl import (
    RLFeatureBuilder,
    SwitcherEnv,
    SwitcherPPO,
)
from robot_nav.models.MARL.switcher.config_loader import load_switcher_config
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
# Suppress IRSim warnings - irsim uses loguru, not standard logging
from loguru import logger
logger.disable("irsim")

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # ---- Environment ----
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "n_robots": 14,
    "n_obstacles": 7,
    "disable_plotting": True,
    "obstacle_proximity_threshold": 1.5,
    "goal_threshold": 0.3,

    # ---- Switcher scalar config (YAML path) ----
    "switcher_config_path": "robot_nav/models/MARL/switcher/switcher_config.yaml",

    "max_episode_steps": 1500,          # sim steps per episode
    "selection_interval": 10,           # sim steps per switcher decision
    "reward_phase": 5,                  # underlying sim reward phase (unused by switcher)
    "per_robot_goal_reset": False,      # do NOT auto-reset individual robots on arrival

    # ---- Decentralized policy (frozen) ----
    "state_dim": 11,
    "obstacle_state_dim": 4,
    # "decentralized_model_name": "TD3-MARL-obstacle-14robots-gpu_epoch800",
    # "decentralized_model_dir": "robot_nav/models/MARL/marlTD3/checkpoint/"
    #                            "Feb.10_obstacle_14robot_transfer_gpu",
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-partial-inactive_epoch210",
    "decentralized_model_dir": "robot_nav/models/MARL/marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive",

    # ---- Group generation ----
    "include_sizes": (2, 3),  # candidate group sizes

    # ---- PPO hyperparameters ----
    "embed_dim": 512,                   # per-robot embedding dim (H from GAT)
    "lr_actor": 5e-5,                   # slow actor LR to preserve pre-trained features
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "eps_clip": 0.15,                   # tighter clipping for stable fine-tuning
    "entropy_coeff": 0.08,             # high initial entropy (annealed down)
    "entropy_coeff_end": 0.01,         # final entropy coefficient
    "entropy_anneal_frac": 0.6,        # anneal over first 60% of max_updates
    "value_coeff": 0.5,
    "max_grad_norm": 1.0,
    "ppo_epochs": 4,                    # fewer epochs to avoid overfitting small buffer
    "embed_hidden": 256,                # hidden dim for embedding tower
    "group_scalar_hidden": 64,
    "fusion_hidden": 256,
    "value_embed_hidden": 128,
    "value_scalar_hidden": 32,

    # ---- Training schedule ----
    "max_updates": 5000,                # 2× updates to match total experience (256×6000 ≈ 500×3000)
    "rollout_steps": 256,               # switcher decisions (NOT sim steps) to collect

    "seed": 42,
    "log_window": 30,                   # wider window for smoother stats

    # =========================================================================
    # Loading Mode
    # =========================================================================
    # 0 = fresh start (no loading)
    # 1 = supervised warm-start: load actor from supervised checkpoint,
    #     freeze actor and warm-up critic for `warmup_updates` updates,
    #     then unfreeze. iter_count=0, optimizer=fresh.
    # 2 = resume training: restore weights + optimizer + iter_count,
    #     entropy annealing continues from where it left off.
    # 3 = warm-start from RL checkpoint: load weights only,
    #     iter_count=0, optimizer=reset, annealing restarts.
    "load_mode": 0,
    "load_checkpoint": "robot_nav/models/MARL/switcher/runs/switcher/epoch_100.pt",              # path or name depending on mode:
    #   mode 1 → full path to supervised .pt file
    #             e.g. "robot_nav/models/MARL/switcher/runs/switcher/epoch_50.pt"
    #   mode 2/3 → checkpoint filename without .pt (empty → use model_name)
    #             e.g. "SwitcherPPO-14robots_update400"
    # "load_directory": "robot_nav/models/MARL/switcher/checkpoint/rl_switcher_14robots",
    "load_directory": "robot_nav/models/MARL/switcher/runs/switcher",  # for mode 2/3
    "warmup_updates": 200,              # mode 1 only: critic warm-up updates

    # ---- Reward coefficients (SwitcherEnv) ----
    # Scaled ~20× down so typical per-step reward ∈ [-0.5, +0.5]
    # and episode returns ∈ [-50, +50].  Keeps value_loss manageable.
    "k_progress": 0.1,                  # dense: ≈0.05-0.4/step for a group
    "k_reach": 5.0,                     # sparse: 5/n_remaining per reach
    "k_all_reached": 25.0,              # sparse: big bonus when ALL reach
    "k_sync": 0.1,                      # dense: variance reduction
    "k_evasion": 0.0,                   # disabled for now (set >0 to enable)
    "collision_penalty": -10.0,
    "time_penalty": -0.1,
    "robot_proximity_threshold": 1.25,
    "obstacle_proximity_threshold": 1.25,

    # ---- Checkpointing ----
    "save_every": 40,                   # save every N PPO updates
    "checkpoint_every": 200,            # save numbered checkpoint every N updates (0 = off)
    "model_name": "SwitcherPPO-14robots",
    "save_directory": "robot_nav/models/MARL/switcher/checkpoint/rl_switcher_14robots/Mar.18",
}


# =============================================================================
# Helpers
# =============================================================================
def _generate_groups(num_robots: int, include_sizes):
    m = 3 if num_robots <= 6 else 4
    all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)
    allowed = set(include_sizes)
    return [g for g in all_groups if len(g) in allowed]


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


# =============================================================================
# Main
# =============================================================================
def main():
    cfg = CONFIG
    _set_seed(cfg["seed"])

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ---- 1. Simulation ----
    sim = MARL_SIM_OBSTACLE(
        world_file=cfg["world_file"],
        disable_plotting=cfg["disable_plotting"],
        reward_phase=cfg["reward_phase"],
        per_robot_goal_reset=cfg["per_robot_goal_reset"],
        obstacle_proximity_threshold=cfg["obstacle_proximity_threshold"],
    )
    print(f"Sim: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
          f"world x={sim.x_range} y={sim.y_range}")

    # ---- 2. Frozen decentralized policy ----
    policy = TD3Obstacle(
        state_dim=cfg["state_dim"],
        action_dim=2,
        max_action=1,
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        obstacle_state_dim=cfg["obstacle_state_dim"],
        device=device,
        load_model=True,
        model_name=cfg["decentralized_model_name"],
        load_model_name=cfg["decentralized_model_name"],
        load_directory=Path(cfg["decentralized_model_dir"]),
        save_directory=Path(cfg["decentralized_model_dir"]),
        inference_only=True,
    )
    policy.actor.eval()
    print(f"Loaded frozen policy: {cfg['decentralized_model_name']}")

    # ---- 3. Groups ----
    groups = _generate_groups(cfg["n_robots"], cfg["include_sizes"])
    print(f"Candidate groups: {len(groups)}  "
          f"(sizes: {sorted(set(len(g) for g in groups))})")

    # ---- 4. Feature builder (always from YAML config) ----
    sw_cfg = load_switcher_config(cfg["switcher_config_path"])
    coupling_mode = sw_cfg.coupling_mode
    group_scalar_dim = sw_cfg.group_scalar_dim
    state_scalar_dim = sw_cfg.state_scalar_dim
    print(f"Switcher config: {cfg['switcher_config_path']}")
    print(f"  coupling_mode={coupling_mode}, "
          f"group_scalar_dim={group_scalar_dim}, state_scalar_dim={state_scalar_dim}")

    fb = RLFeatureBuilder.from_config(
        sw_cfg,
        embed_dim=cfg["embed_dim"],
        pooling="mean",
        max_group_size=max(len(g) for g in groups),
    )

    # ---- 5. Switcher environment ----
    env = SwitcherEnv(
        sim=sim,
        policy=policy,
        groups=groups,
        feature_builder=fb,
        selection_interval=cfg["selection_interval"],
        max_episode_steps=cfg["max_episode_steps"],
        goal_threshold=cfg["goal_threshold"],
        device=device_str,
        coupling_mode=coupling_mode,
    )
    # Override reward coefficients
    env.k_progress = cfg["k_progress"]
    env.k_reach = cfg["k_reach"]
    env.k_all_reached = cfg["k_all_reached"]
    env.k_sync = cfg["k_sync"]
    env.k_evasion = cfg["k_evasion"]
    env.collision_penalty = cfg["collision_penalty"]
    env.time_penalty = cfg["time_penalty"]
    env.robot_proximity_threshold = cfg["robot_proximity_threshold"]
    env.obstacle_proximity_threshold = cfg["obstacle_proximity_threshold"]

    # Enable reward debugging
    env.debug_rewards = False

    # ---- 6. PPO agent ----
    ppo = SwitcherPPO(
        embed_dim=cfg["embed_dim"],
        group_scalar_dim=group_scalar_dim,
        state_scalar_dim=state_scalar_dim,
        lr_actor=cfg["lr_actor"],
        lr_critic=cfg["lr_critic"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        eps_clip=cfg["eps_clip"],
        entropy_coeff=cfg["entropy_coeff"],
        entropy_coeff_end=cfg.get("entropy_coeff_end"),
        entropy_anneal_updates=int(cfg.get("entropy_anneal_frac", 0) * cfg["max_updates"]),
        value_coeff=cfg["value_coeff"],
        max_grad_norm=cfg["max_grad_norm"],
        device=device_str,
        save_every=cfg["save_every"],
        model_name=cfg["model_name"],
        save_directory=Path(cfg["save_directory"]),
        # Network architecture kwargs
        embed_hidden=cfg["embed_hidden"],
        group_scalar_hidden=cfg["group_scalar_hidden"],
        fusion_hidden=cfg["fusion_hidden"],
        value_embed_hidden=cfg["value_embed_hidden"],
        value_scalar_hidden=cfg["value_scalar_hidden"],
    )
    # Attach switcher config for checkpoint persistence
    ppo.switcher_config_dict = sw_cfg.to_dict()

    # ---- Apply loading mode ----
    load_mode = cfg.get("load_mode", 0)
    load_checkpoint = cfg.get("load_checkpoint", "") or cfg["model_name"]
    load_dir = Path(cfg["load_directory"])

    if load_mode not in (0, 1, 2, 3):
        raise ValueError(f"load_mode must be 0, 1, 2, or 3 (got {load_mode})")

    if load_mode == 1:
        # Supervised warm-start: load actor weights, warm-up critic, then unfreeze
        ppo.load_supervised_weights(cfg["load_checkpoint"])
        warmup_updates = cfg.get("warmup_updates", 0)
        if warmup_updates > 0:
            print(f"{'=' * 60}")
            print(f"[Mode 1] Critic warm-up: {warmup_updates} updates (actor frozen)")
            print(f"{'=' * 60}")
            ppo.freeze_actor()
            wu_gf, wu_sf = env.reset()
            wu_episodes = 0
            for wu_update in range(1, warmup_updates + 1):
                for _ in range(cfg["rollout_steps"]):
                    group_idx = ppo.get_action(wu_gf, wu_sf, explore=True)
                    wu_gf, wu_sf, reward, done, info = env.step(group_idx)
                    ppo.store_reward(reward, done)
                    if done:
                        wu_episodes += 1
                        wu_gf, wu_sf = env.reset()
                ppo.train(iterations=cfg["ppo_epochs"])
                if wu_update % 20 == 0 or wu_update == warmup_updates:
                    print(f"  Warm-up {wu_update:4d}/{warmup_updates}  episodes={wu_episodes}")
            ppo.unfreeze_actor()
            ppo.iter_count = 0
            ppo.save(filename=f"{cfg['model_name']}_warmup", directory=Path(cfg["save_directory"]))
            print(f"Critic warm-up complete — actor unfrozen\n")

    elif load_mode == 2:
        # Resume: restore weights + optimizer + iter_count
        ppo.load(filename=load_checkpoint, directory=load_dir)
        print(f"[Mode 2] Resumed from: {load_dir}/{load_checkpoint}.pt  "
              f"(iter_count={ppo.iter_count})")

    elif load_mode == 3:
        # Warm-start from RL checkpoint: weights only, fresh optimizer & iter_count
        ppo.load_weights_only(filename=load_checkpoint, directory=load_dir)
        print(f"[Mode 3] Warm-started from: {load_dir}/{load_checkpoint}.pt")

    # ---- Print summary ----
    n_params = sum(p.numel() for p in ppo.policy.parameters())
    mode_labels = {
        0: "fresh start",
        1: "supervised warm-start",
        2: "resume training",
        3: "RL warm-start",
    }
    print(f"\nSwitcherPPO parameters: {n_params:,}")
    print(f"Load mode: {load_mode} ({mode_labels[load_mode]})")
    print(f"Group feature dim (actor input per group): {fb.group_feature_dim}")
    print(f"State feature dim (critic input): {fb.state_feature_dim}")
    print(f"Rollout steps per update: {cfg['rollout_steps']}")
    print(f"PPO epochs per update: {cfg['ppo_epochs']}")
    print(f"Max updates: {cfg['max_updates']}")
    if cfg.get("entropy_coeff_end") is not None:
        anneal_updates = int(cfg.get("entropy_anneal_frac", 0) * cfg["max_updates"])
        print(f"Entropy annealing: {cfg['entropy_coeff']} → {cfg['entropy_coeff_end']} "
              f"over {anneal_updates} updates")
    print()

    # ---- 7. Training loop ----
    # Rolling statistics
    ep_rewards = deque(maxlen=cfg["log_window"])
    ep_lengths = deque(maxlen=cfg["log_window"])
    ep_reached = deque(maxlen=cfg["log_window"])
    ep_collisions = deque(maxlen=cfg["log_window"])
    ep_all_reached = deque(maxlen=cfg["log_window"])

    total_episodes = 0
    total_decisions = 0
    update_count = 0
    t_start = time.time()

    # Start first episode
    group_features, state_features = env.reset()
    ep_reward = 0.0
    ep_len = 0

    while update_count < cfg["max_updates"]:
        # ---- Collect rollout: rollout_steps switcher decisions ----
        # Each decision = 1 PPO action + 10 sim steps inside env.step()
        for _ in range(cfg["rollout_steps"]):
            # Select group via PPO (stores obs/logprob/value in buffer)
            group_idx = ppo.get_action(group_features, state_features, explore=True)

            # Step environment for selection_interval sim steps
            group_features, state_features, reward, done, info = env.step(group_idx)

            # Store reward + terminal in PPO buffer
            ppo.store_reward(reward, done)

            ep_reward += reward
            ep_len += 1
            total_decisions += 1

            if done:
                # Episode finished — log stats
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_reached.append(info["n_reached"])
                ep_collisions.append(1 if info["collision"] or info["oob"] else 0)
                ep_all_reached.append(1 if info["all_reached"] else 0)
                total_episodes += 1

                # Reset for next episode
                group_features, state_features = env.reset()
                ep_reward = 0.0
                ep_len = 0

        # ---- PPO update ----
        ppo.train(iterations=cfg["ppo_epochs"])
        update_count += 1

        # ---- Numbered checkpoint ----
        checkpoint_every = cfg.get("checkpoint_every", 0)
        if checkpoint_every > 0 and update_count % checkpoint_every == 0:
            ckpt_name = f"{cfg['model_name']}_update{update_count}"
            ppo.save(filename=ckpt_name, directory=Path(cfg["save_directory"]))
            print(f"✅ Checkpoint saved: {ckpt_name}")

        # ---- Logging ----
        if update_count % 1 == 0 and len(ep_rewards) > 0:
            elapsed = time.time() - t_start
            avg_r = np.mean(ep_rewards)
            avg_len = np.mean(ep_lengths)
            avg_reached = np.mean(ep_reached)
            col_rate = np.mean(ep_collisions)
            all_rate = np.mean(ep_all_reached)

            print(
                f"[Update {update_count:4d}] "
                f"ep={total_episodes:5d}  "
                f"dec={total_decisions:7d}  "
                f"R={avg_r:+7.1f}  "
                f"len={avg_len:5.1f}  "
                f"reached={avg_reached:.1f}/{cfg['n_robots']}  "
                f"col={col_rate:.2f}  "
                f"all_reach={all_rate:.2f}  "
                f"t={elapsed:.0f}s"
            )

            # TensorBoard logging (via PPO writer)
            ppo.writer.add_scalar("rollout/mean_reward", avg_r, update_count)
            ppo.writer.add_scalar("rollout/mean_episode_length", avg_len, update_count)
            ppo.writer.add_scalar("rollout/mean_reached", avg_reached, update_count)
            ppo.writer.add_scalar("rollout/collision_rate", col_rate, update_count)
            ppo.writer.add_scalar("rollout/all_reached_rate", all_rate, update_count)

    # ---- Final save ----
    ppo.save(filename=cfg["model_name"], directory=Path(cfg["save_directory"]))
    print(f"\nTraining complete. Final checkpoint saved to {cfg['save_directory']}")
    print(f"Total episodes: {total_episodes}, total decisions: {total_decisions}")


if __name__ == "__main__":
    main()
