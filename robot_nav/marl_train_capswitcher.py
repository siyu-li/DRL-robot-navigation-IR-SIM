"""
Training script for CAPSwitcher PPO.

Trains a binary switcher (coarse vs. precise) on top of a frozen GAT backbone
(pre-trained TD3Obstacle for 6-robot obstacle navigation).  The switcher learns
to select coarse group control in free space and precise individual control near
obstacles, producing more physically efficient navigation than either mode alone.

Usage:
    python -m robot_nav.marl_train_capswitcher

Key hyperparameters
-------------------
  rollout_steps  : 512   — switcher decisions per PPO update
  ppo_epochs     : 10    — PPO mini-batch passes per update
  batch_size     : 64
  selection_interval : 5 — sim steps committed per switcher decision
  mode costs     : coarse=-0.2/step, precise=-1.0/step
  max_updates    : 10000
"""

from pathlib import Path

import torch
from loguru import logger

from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv
from robot_nav.models.MARL.capswitcher.rl.switcher_ppo import SwitcherPPO

# Suppress irsim logging noise
logger.disable("irsim")


def main() -> None:
    # ------------------------------------------------------------------ #
    # Device                                                               #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    )

    # ------------------------------------------------------------------ #
    # Coarse steering                                                      #
    # ------------------------------------------------------------------ #
    coarse_steering = CoarseSteering(
        num_robots=sim.num_robots,
        use_groups=2,   # rank-2 actuation matrix
        lin_vel=0.25,   # moderate forward speed in sim-input units
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
    # PPO trainer                                                          #
    # ------------------------------------------------------------------ #
    ppo = SwitcherPPO(
        embed_dim=512,
        hidden1=256,
        hidden2=128,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ppo_epochs=10,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        writer_comment="CAPSwitcher",
        save_dir=Path(
            "robot_nav/models/MARL/capswitcher/checkpoints/cap_switcher"
        ),
        model_name="capswitcher",
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    num_updates = 10_000
    rollout_steps = 512
    save_every = 100

    print(
        f"\nStarting CAPSwitcher PPO training for {num_updates} updates ...\n"
        f"  rollout_steps={rollout_steps}, ppo_epochs={ppo.ppo_epochs}, "
        f"batch_size={ppo.batch_size}, selection_interval={env.selection_interval}"
    )

    for update in range(1, num_updates + 1):
        # ---- Collect rollout -------------------------------------------
        buffer, rollout_stats = ppo.collect_rollout(env, rollout_steps=rollout_steps)

        # Bootstrap value from the final observation after the rollout.
        last_obs = env._get_obs()

        # ---- PPO update ------------------------------------------------
        loss_stats = ppo.train(buffer, last_obs=last_obs)

        ppo.update_count = update

        # ---- TensorBoard -----------------------------------------------
        ppo.log(rollout_stats, loss_stats)

        # ---- Console ---------------------------------------------------
        if update % 10 == 0:
            print(
                f"Update {update:5d}/{num_updates} | "
                f"Reward: {rollout_stats['mean_reward']:+7.2f} | "
                f"Coarse%: {rollout_stats['frac_coarse']*100:5.1f}% | "
                f"Success: {rollout_stats['success_rate']*100:5.1f}% | "
                f"Collision: {rollout_stats['collision_rate']*100:5.1f}% | "
                f"PiLoss: {loss_stats['ppo/policy_loss']:.4f} | "
                f"VLoss: {loss_stats['ppo/value_loss']:.4f} | "
                f"Entropy: {loss_stats['ppo/entropy']:.4f}"
            )

        # ---- Checkpoint ------------------------------------------------
        if update % save_every == 0:
            ppo.save()

    print("\nTraining complete!")
    ppo.save(filename=f"{ppo.model_name}_final")


if __name__ == "__main__":
    main()
