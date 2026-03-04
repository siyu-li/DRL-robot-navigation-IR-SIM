"""
Training script for Learned Group Mixing — 14 Robots.

Freezes the pre-trained GAT + TD3 policy entirely and trains **only** a small
shared ``MixingNetwork`` that maps per-robot embeddings to softmax mixing
weights.  The mixed velocity is applied to all robots in the active group.

Key design choices
------------------
* The **frozen critic** is the reward.  No new reward function is needed.
  The critic already encodes proximity, collision, and progress penalties
  from the original training.  ``loss = −Q_frozen(s, a_coupled)`` tells the
  mixing network whether its weight distribution produced a good coupled
  action.
* Gradient path:
      −Q → a_coupled → v_shared = Σ α_i · v_i → α = softmax(f_lin(e_i)) → θ_flin
  Only ``f_lin`` parameters receive gradients.
* Data collection mirrors the original TD3 loop: rollout with the frozen
  actor + mixing network, store transitions in a replay buffer, sample
  batches off-policy, and run the critic-based update.
* Every ``group_interval`` env steps a new group **G** is sampled uniformly
  from all size-2 and size-3 combinatorial groups.

Usage
-----
    python -m robot_nav.marl_train_group_mixing_14robots
"""

from pathlib import Path
import random

import torch
import torch.nn.functional as F
import numpy as np
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.marlTD3.replay_buffer_obstacle import ReplayBufferObstacle
from robot_nav.models.MARL.groups.mixing_network import MixingNetwork
from robot_nav.models.MARL.groups.combinatorial_group_generator import (
    generate_all_combinations,
    print_combination_statistics,
)
from robot_nav.models.MARL.groups.learned_action_coupling import (
    compute_mixed_actions,
    compute_mixed_actions_tensor,
    get_embeddings_from_frozen_actor,
)

from loguru import logger
logger.disable("irsim")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def outside_of_bounds(poses, sim):
    """Check if any robot is outside world boundaries."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Train the MixingNetwork while keeping GAT + TD3 frozen."""

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Hyper-parameters ----
    state_dim = 11
    action_dim = 2
    max_action = 1
    obstacle_state_dim = 4
    embedding_dim = 256  # matches ActorObstacle / AttentionObstacleOptimized

    # Training
    max_epochs = 3000
    max_steps = 300
    group_interval = 20        # re-sample a group every N env steps
    lr_mixing = 3e-4           # learning rate for MixingNetwork
    max_grad_norm = 5.0
    save_every = 200           # checkpoint every N epochs
    train_every_n = 10         # train after this many episodes
    training_iterations = 40   # gradient steps per training round
    batch_size = 32
    buffer_size = 100000

    # Group generation (all combinatorial groups for 14 robots, m=4)
    max_group_size = 3         # only size-2 and size-3 groups
    min_group_size = 2         # only groups with ≥ 2 robots need mixing

    # Environment uses the SAME reward phase as the frozen policy was trained
    # on.  The env reward is stored in the buffer but NOT used for the mixing
    # network update — only the frozen critic's Q-value matters.
    reward_phase = 8

    # Dwell
    goal_dwell_min = 0
    goal_respawn_prob = 1.0
    station_keeping_reward = 5.0

    # Directories
    frozen_model_dir = Path(
        "robot_nav/models/MARL/marlTD3/checkpoint/Mar.02_obstacle_14robot_reward8"
    )
    frozen_model_name = "TD3-MARL-obstacle-14robots"
    save_directory = Path(
        "robot_nav/models/MARL/groups/checkpoint/group_mixing_14robots"
    )
    save_directory.mkdir(parents=True, exist_ok=True)

    # ---- Environment ----
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
        disable_plotting=True,
        reward_phase=reward_phase,
        per_robot_goal_reset=True,
        obstacle_proximity_threshold=1.5,
        goal_dwell_min=goal_dwell_min,
        goal_respawn_prob=goal_respawn_prob,
        station_keeping_reward=station_keeping_reward,
    )
    num_robots = sim.num_robots
    num_obstacles = sim.num_obstacles
    print(f"Environment: {num_robots} robots, {num_obstacles} obstacles")

    # ---- Frozen policy (actor + critic) ----
    frozen_policy = TD3Obstacle(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=num_robots,
        num_obstacles=num_obstacles,
        obstacle_state_dim=obstacle_state_dim,
        device=device,
        load_model=True,
        load_model_name=frozen_model_name,
        load_directory=frozen_model_dir,
        inference_only=True,
    )
    # Freeze everything
    for p in frozen_policy.actor.parameters():
        p.requires_grad = False
    for p in frozen_policy.critic.parameters():
        p.requires_grad = False
    frozen_policy.actor.eval()
    frozen_policy.critic.eval()
    print("Frozen policy loaded and frozen (actor + critic).")

    # ---- Mixing network (the ONLY trainable thing) ----
    mixing_net = MixingNetwork(
        embedding_dim=embedding_dim * 2,  # actor attention output is embed_dim*2
        hidden_dim=128,
    ).to(device)
    mixing_optimizer = torch.optim.Adam(mixing_net.parameters(), lr=lr_mixing)
    print(f"MixingNetwork parameters: {sum(p.numel() for p in mixing_net.parameters()):,}")

    # ---- Group catalogue (all C(14,2) + C(14,3) combinations) ----
    groups = generate_all_combinations(
        n=num_robots, min_size=min_group_size, max_size=max_group_size
    )
    print_combination_statistics(groups, n=num_robots, max_display_per_size=10)
    print(f"Using {len(groups)} groups (size {min_group_size}–{max_group_size})")

    # ---- Replay buffer ----
    replay_buffer = ReplayBufferObstacle(buffer_size=buffer_size)

    # ---- Logging ----
    writer = SummaryWriter(comment="GroupMixing-14robots")

    # ---- Initial env step ----
    (
        poses, distance, cos_val, sin_val, collision, goal, a, reward,
        positions, goal_positions, obstacle_states
    ) = sim.step([[0, 0]] * num_robots, None)

    # ---- Bookkeeping ----
    epoch = 0
    episode = 0
    steps = 0
    iter_count = 0
    running_goals = 0
    running_collisions = 0
    running_timesteps = 0
    checkpoint_every = 200

    # Track which group is active (for the buffer — we store the coupled
    # actions, not the raw ones, so the critic sees what was actually executed)
    active_group = random.choice(groups)

    print(f"\nStarting training for {max_epochs} epochs …")
    print("=" * 70)

    # ================================================================
    #  DATA COLLECTION + TRAINING LOOP  (mirrors marl_train_obstacle)
    # ================================================================
    while epoch < max_epochs:

        # ---- Prepare robot state ----
        robot_state, terminal = frozen_policy.prepare_state(
            poses, distance, cos_val, sin_val, collision, a, goal_positions
        )
        robot_obs = np.array(robot_state)

        # ---- Every group_interval steps, pick a new group ----
        if steps % group_interval == 0:
            active_group = random.choice(groups)

        # ---- Frozen actor → raw actions + embeddings (no grad) ----
        with torch.no_grad():
            raw_action, combined_weights = frozen_policy.get_action(
                robot_obs, obstacle_states, add_noise=False
            )

        embeddings = get_embeddings_from_frozen_actor(
            frozen_policy.actor, robot_obs, obstacle_states, device
        )

        # ---- Arrived mask ----
        arrived_mask = [c >= 0 for c in sim.dwell_counters]

        # ---- Mixing network → coupled actions for env (no grad) ----
        a_in = compute_mixed_actions(
            mixing_net=mixing_net,
            embeddings=embeddings,
            raw_actions=raw_action,
            group=active_group,
            num_robots=num_robots,
            arrived_mask=arrived_mask,
            rotation_coupling_threshold=3,
            scale_by_sqrt=True,
        )

        # ---- Step environment ----
        (
            poses, distance, cos_val, sin_val, collision, goal, a, reward,
            positions, goal_positions, next_obstacle_states
        ) = sim.step(a_in, None, combined_weights)

        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_timesteps += 1

        # ---- Prepare next state ----
        next_robot_state, terminal = frozen_policy.prepare_state(
            poses, distance, cos_val, sin_val, collision, a, goal_positions
        )

        # ---- Store transition in replay buffer ----
        # We store the RAW actions (not scaled) so the critic can evaluate them.
        # The raw_action is what the frozen actor produced; the coupling is
        # re-applied differentiably during the training step.
        replay_buffer.add(
            robot_state,
            obstacle_states,
            raw_action,        # store raw [-1,1] actions
            reward,
            terminal,
            next_robot_state,
            next_obstacle_states,
            active_mask=sim.active_mask,
        )

        obstacle_states = next_obstacle_states
        steps += 1
        episode += 1

        # ---- Episode termination ----
        if any(collision) or steps >= max_steps or outside_of_bounds(poses, sim):
            (
                poses, distance, cos_val, sin_val, collision, goal, a, reward,
                positions, goal_positions, obstacle_states
            ) = sim.reset(random_obstacles=True)

            steps = 0
            epoch += 1
            active_group = random.choice(groups)

            # ========================================================
            #  TRAINING STEP — TD3-style, frozen critic as evaluator
            # ========================================================
            if episode >= train_every_n and replay_buffer.size() >= batch_size:

                # Log run metrics
                avg_goal_rate = running_goals / max(running_timesteps, 1)
                avg_collision_rate = running_collisions / max(running_timesteps, 1)
                writer.add_scalar("run/avg_goal", avg_goal_rate, iter_count)
                writer.add_scalar("run/avg_collision", avg_collision_rate, iter_count)
                writer.add_scalar("run/buffer_size", replay_buffer.size(), iter_count)
                running_goals = 0
                running_collisions = 0
                running_timesteps = 0

                av_mixing_loss = 0.0
                av_Q = 0.0
                max_Q = -inf

                for it in range(training_iterations):
                    # ---- Sample batch from replay buffer ----
                    (
                        batch_robot_states,
                        batch_obstacle_states,
                        batch_actions,       # raw actions from frozen actor
                        batch_rewards,       # NOT used for mixing update
                        batch_dones,
                        batch_next_robot_states,
                        batch_next_obstacle_states,
                        batch_active_masks,
                    ) = replay_buffer.sample_batch(batch_size)

                    robot_state_t = (
                        torch.Tensor(batch_robot_states).to(device)
                        .view(batch_size, num_robots, state_dim)
                    )
                    obstacle_state_t = (
                        torch.Tensor(batch_obstacle_states).to(device)
                        .view(batch_size, num_obstacles, obstacle_state_dim)
                    )
                    raw_actions_t = (
                        torch.Tensor(batch_actions).to(device)
                        .view(batch_size, num_robots, action_dim)
                    )
                    active_mask_t = (
                        torch.Tensor(batch_active_masks.astype(float)).to(device)
                        .view(batch_size * num_robots, 1)
                    )

                    # ---- For each sample in the batch: ----
                    # 1. Get frozen embeddings
                    # 2. Pick a random group
                    # 3. Run mixing network differentiably → coupled action
                    # 4. Evaluate with frozen critic → Q
                    # 5. loss = −Q (averaged over group members)

                    # Get frozen embeddings for entire batch
                    with torch.no_grad():
                        (
                            batch_embeddings,  # (B*N, embed_dim*2)
                            _, _, _, _, _, _, _, _,
                        ) = frozen_policy.actor.attention(
                            robot_state_t, obstacle_state_t
                        )
                    batch_embeddings = batch_embeddings.detach().view(
                        batch_size, num_robots, -1
                    )  # (B, N, embed_dim*2)

                    # Build coupled actions for entire batch
                    # Each sample gets a randomly chosen group
                    all_coupled_actions = []
                    group_masks = []  # which robots are in the group per sample

                    for b in range(batch_size):
                        grp = random.choice(groups)
                        coupled = compute_mixed_actions_tensor(
                            mixing_net=mixing_net,
                            embeddings=batch_embeddings[b],       # (N, embed_dim*2)
                            raw_actions_tensor=raw_actions_t[b],  # (N, 2)
                            group=grp,
                            num_robots=num_robots,
                            arrived_mask=None,  # no dwell info in buffer
                            rotation_coupling_threshold=3,
                            scale_by_sqrt=True,
                        )
                        all_coupled_actions.append(coupled)

                        # Mask: 1 for group members, 0 for others
                        gmask = torch.zeros(num_robots, 1, device=device)
                        for idx in grp:
                            gmask[idx, 0] = 1.0
                        group_masks.append(gmask)

                    # Stack → (B, N, 2) and (B*N, 2)
                    coupled_actions_t = torch.stack(all_coupled_actions)  # (B, N, 2)
                    coupled_flat = coupled_actions_t.view(batch_size * num_robots, action_dim)

                    group_mask_t = torch.cat(group_masks, dim=0)  # (B*N, 1)

                    # ---- Frozen critic evaluates the coupled action ----
                    # Critic params are frozen (requires_grad=False) so no
                    # gradient accumulates there, but the graph through
                    # coupled_flat → α → f_lin IS kept for backward().
                    Q1, _, _, _, _, _, _, _, _ = frozen_policy.critic(
                        robot_state_t, obstacle_state_t, coupled_flat
                    )

                    # Track Q stats
                    with torch.no_grad():
                        av_Q += Q1.mean().item()
                        max_Q = max(max_Q, Q1.max().item())

                    # ---- Loss = −Q for group members only ----
                    mixing_loss = -(Q1 * group_mask_t).sum() / group_mask_t.sum()

                    mixing_optimizer.zero_grad()
                    mixing_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        mixing_net.parameters(), max_grad_norm
                    )
                    mixing_optimizer.step()

                    av_mixing_loss += mixing_loss.item()

                iter_count += 1
                episode = 0

                # ---- Logging ----
                writer.add_scalar(
                    "train/mixing_loss",
                    av_mixing_loss / training_iterations,
                    iter_count,
                )
                writer.add_scalar(
                    "train/avg_Q", av_Q / training_iterations, iter_count
                )
                writer.add_scalar("train/max_Q", max_Q, iter_count)

                # ---- Checkpointing ----
                if epoch % checkpoint_every == 0:
                    ckpt_path = save_directory / f"mixing_net_epoch{epoch}.pth"
                    torch.save(mixing_net.state_dict(), ckpt_path)
                    print(f"✅ Checkpoint saved: {ckpt_path}")

                # ---- Console logging ----
                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch}/{max_epochs} | "
                        f"Buffer: {replay_buffer.size()} | "
                        f"MixLoss: {av_mixing_loss / training_iterations:.4f} | "
                        f"AvgQ: {av_Q / training_iterations:.2f} | "
                        f"Goals: {avg_goal_rate*100:.1f}% | "
                        f"Collisions: {avg_collision_rate*100:.1f}%"
                    )

    # ---- Final save ----
    final_path = save_directory / "mixing_net_final.pth"
    torch.save(mixing_net.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
