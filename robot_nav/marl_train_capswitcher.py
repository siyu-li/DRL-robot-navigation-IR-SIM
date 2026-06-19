"""
Training script for CAPSwitcher (off-policy DQN).

Trains a binary switcher (coarse vs. precise) on top of a frozen GAT backbone
(pre-trained TD3Obstacle for 6-robot obstacle navigation).  The switcher learns
to select coarse group control in free space and precise individual control near
obstacles, producing more physically efficient navigation than either mode alone.

Why DQN (off-policy) instead of PPO
-----------------------------------
The switcher is a binary decision over a frozen backbone observed as a fixed
512-dim embedding.  A replay buffer lets every collected transition be reused
for many gradient updates, so far fewer (expensive) IR-SIM steps are needed to
converge than with on-policy PPO, which discards each rollout.  The cached
embeddings live in the buffer, so DQN updates are pure MLP passes — the GAT
backbone runs only during data collection.

Device note (Plan A)
---------------------
The backbone forward during rollout is batch-1 over a ~6-node graph.  On such a
tiny graph CPU is faster than CUDA (kernel-launch + host<->device transfer
latency dominates the matmul), so the environment/backbone run on CPU
deliberately.  The DQN MLP is also tiny and runs on CPU.

Usage:
    python -m robot_nav.marl_train_capswitcher
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv
from robot_nav.models.MARL.capswitcher.rl.switcher_dqn import SwitcherDQN

# Suppress irsim logging noise
logger.disable("irsim")


def main() -> None:
    # ------------------------------------------------------------------ #
    # Device — keep the rollout backbone + DQN on CPU (see module docstring) #
    # ------------------------------------------------------------------ #
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Simulation environment                                               #
    # ------------------------------------------------------------------ #
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle.yaml",
        disable_plotting=True,
        reward_phase=6,          # progress + collision + goal
        per_robot_goal_reset=True,
        obstacle_proximity_threshold=1.5,
        num_inactive_robots=0,
    )
    print(
        f"Environment ready: {sim.num_robots} robots, "
        f"{sim.num_obstacles} obstacles, "
        f"world x={sim.x_range} y={sim.y_range}"
    )

    # ------------------------------------------------------------------ #
    # Frozen GAT backbone                                                  #
    # ------------------------------------------------------------------ #
    gat_backbone = GATBackbone(
        checkpoint_path=Path(
            "robot_nav/models/MARL/marlTD3/checkpoint/"
            "obstacle_6robots_v4/TD3-MARL-obstacle-6robots-reward6"
        ),
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        device=device,
        embedding_source="decoder",  # per-robot decoder output (attn_out)
    )

    # ------------------------------------------------------------------ #
    # Coarse steering                                                      #
    # ------------------------------------------------------------------ #
    coarse_steering = CoarseSteering(
        num_robots=sim.num_robots,
        lin_vel=0.25,                # moderate forward speed in sim-input units
        method="least_squares",      # or "nonlinear" (location-progress optimiser)
        step_time=sim.env.step_time,  # used to fully realise rotations sub-step-wise
        ang_max=1.0,                 # sim angular-velocity bound
    )

    # ------------------------------------------------------------------ #
    # Switcher environment                                                 #
    # ------------------------------------------------------------------ #
    env = SwitcherEnv(
        sim=sim,
        backbone=gat_backbone,
        coarse_steering=coarse_steering,
        selection_interval=5,
        max_steps=300,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # DQN trainer                                                          #
    # ------------------------------------------------------------------ #
    dqn = SwitcherDQN(
        num_robots=sim.num_robots,
        embed_dim=env.EMBED_DIM,        # 512 per-robot decoder output
        phi_dims=(256, 128),            # Deep Sets per-robot encoder φ
        rho_dims=(128,),                # post-aggregation ρ
        aggregation="sum_max",          # sum (density) ⊕ max (worst-case robot)
        num_actions=env.NUM_ACTIONS,
        lr=3e-4,
        gamma=0.99,
        buffer_size=50_000,             # (N,512) obs ≈ 24 KB/transition → ~1.2 GB
        batch_size=128,
        learn_starts=1_000,
        train_freq=1,
        target_update_freq=500,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
        double_dqn=True,
        device=device,
        writer_comment="CAPSwitcherDQN",
        save_dir=Path(
            "robot_nav/models/MARL/capswitcher/checkpoints/cap_switcher"
        ),
        model_name="capswitcher_dqn",
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    total_steps = 200_000   # switcher decisions
    log_every = 1_000       # env steps between TensorBoard writes
    save_every = 20_000     # env steps between checkpoints
    stats_window = 50       # episodes to average for reporting

    print(
        f"\nStarting CAPSwitcher DQN training for {total_steps} steps ...\n"
        f"  batch_size={dqn.batch_size}, buffer={dqn.buffer.capacity}, "
        f"learn_starts={dqn.learn_starts}, target_update={dqn.target_update_freq}, "
        f"selection_interval={env.selection_interval}"
    )

    ep_rewards: deque[float] = deque(maxlen=stats_window)
    ep_lengths: deque[int] = deque(maxlen=stats_window)
    ep_success: deque[float] = deque(maxlen=stats_window)
    ep_collision: deque[float] = deque(maxlen=stats_window)
    ep_coarse_frac: deque[float] = deque(maxlen=stats_window)

    obs = env.reset()
    ep_reward = 0.0
    ep_len = 0
    ep_coarse = 0
    last_loss: dict = {}

    for step in range(1, total_steps + 1):
        eps = dqn.epsilon(step)
        action = dqn.select_action(obs, eps)
        next_obs, reward, done, info = env.step(action)

        dqn.store(obs, action, reward, next_obs, done)

        ep_reward += reward
        ep_len += 1
        if action == 0:
            ep_coarse += 1
        obs = next_obs

        # ---- Learn -----------------------------------------------------
        if step >= dqn.learn_starts and step % dqn.train_freq == 0:
            stats = dqn.update()
            if stats is not None:
                last_loss = stats
        if step % dqn.target_update_freq == 0:
            dqn.update_target()

        # ---- Episode boundary -----------------------------------------
        if done:
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            ep_success.append(1.0 if info.get("all_reached") else 0.0)
            ep_collision.append(1.0 if info.get("collision") else 0.0)
            ep_coarse_frac.append(ep_coarse / max(ep_len, 1))
            ep_reward, ep_len, ep_coarse = 0.0, 0, 0
            obs = env.reset()

        # ---- Logging ---------------------------------------------------
        if step % log_every == 0 and ep_rewards:
            log_stats = {
                "rollout/mean_reward":    float(np.mean(ep_rewards)),
                "rollout/mean_ep_length": float(np.mean(ep_lengths)),
                "switcher/frac_coarse":   float(np.mean(ep_coarse_frac)),
                "episode/success_rate":   float(np.mean(ep_success)),
                "episode/collision_rate": float(np.mean(ep_collision)),
                "train/epsilon":          eps,
                **last_loss,
            }
            dqn.log(step, log_stats)
            print(
                f"Step {step:7d}/{total_steps} | "
                f"Reward: {log_stats['rollout/mean_reward']:+7.2f} | "
                f"Coarse%: {log_stats['switcher/frac_coarse']*100:5.1f}% | "
                f"Success: {log_stats['episode/success_rate']*100:5.1f}% | "
                f"Collision: {log_stats['episode/collision_rate']*100:5.1f}% | "
                f"Eps: {eps:.3f} | "
                f"Loss: {last_loss.get('dqn/loss', float('nan')):.4f}"
            )

        # ---- Checkpoint ------------------------------------------------
        if step % save_every == 0:
            dqn.save(filename=f"{dqn.model_name}_step{step}")

    print("\nTraining complete!")
    dqn.save(filename=f"{dqn.model_name}_final")


if __name__ == "__main__":
    main()
