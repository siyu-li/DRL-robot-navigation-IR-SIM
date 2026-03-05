"""
Training script for Learned Group Mixing — 14 Robots (REINFORCE).

Freezes the pre-trained GAT + TD3 policy entirely and trains **only** a small
shared ``MixingNetwork`` that maps per-robot embeddings to softmax mixing
weights.  The mixed velocity is applied to all robots in the active group.

Key design choices
------------------
* **REINFORCE with Gaussian noise on logits.**  The mixing network produces
  deterministic logits; we add isotropic Gaussian noise to make the policy
  stochastic, then use the score-function estimator.
* **10-step rollouts.**  Each rollout fixes the mixing weights α for 10
  environment steps, re-querying the frozen actor at each step for fresh
  per-robot velocities.
* **Reward phase 9** (proximity + collision only) — no progress reward.
  The reward is averaged over group size so that size-2 and size-3 groups
  are on comparable scales.
* **Separate EMA baselines** per group size for variance reduction.
* **Batch of rollouts** (default 16) per gradient update:
  16 rollouts × 10 steps = 160 env steps per update.

Gradient signal
---------------
    loss = −log π(α | e) · (R̄ − b)
where
    R̄  = (Σ γ^t r_t) / |G|        (average return over group size)
    b  = EMA baseline for the group size
    log π = −0.5 Σ ((noisy_logit_i − logit_i)² / σ²)
           (logits come from f_lin(e_i); noisy_logits are detached targets)

Usage
-----
    python -m robot_nav.marl_train_group_mixing_14robots
"""

from pathlib import Path
import math
import random

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.groups.mixing_network import MixingNetwork
from robot_nav.models.MARL.groups.combinatorial_group_generator import (
    generate_all_combinations,
    print_combination_statistics,
)
from robot_nav.models.MARL.groups.learned_action_coupling import (
    compute_mixed_actions,
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
    """Train the MixingNetwork via REINFORCE while keeping GAT + TD3 frozen."""

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Hyper-parameters ----
    state_dim = 11
    action_dim = 2
    max_action = 1
    obstacle_state_dim = 4
    embedding_dim = 256

    # REINFORCE
    max_updates = 10000        # total gradient updates
    rollout_horizon = 10       # env steps per rollout
    batch_size = 16            # rollouts per gradient update
    lr_mixing = 3e-4
    gamma = 0.99               # discount factor
    max_grad_norm = 5.0
    logit_noise_std = 0.5      # σ for Gaussian noise on logits
    noise_std_min = 0.1        # minimum noise after decay
    noise_decay = 0.9999       # per-update multiplicative decay
    baseline_momentum = 0.99   # EMA momentum for baseline

    # Group generation
    max_group_size = 3
    min_group_size = 2

    # Episode management (reset env periodically to avoid stale states)
    max_steps_before_reset = 2000

    # Reward phase 9 — proximity + collision only
    reward_phase = 9

    # Dwell
    goal_dwell_min = 0
    goal_respawn_prob = 1.0
    station_keeping_reward = 5.0

    # Checkpointing
    checkpoint_every = 200
    save_directory = Path(
        "robot_nav/models/MARL/groups/checkpoint/group_mixing_14robots"
    )
    save_directory.mkdir(parents=True, exist_ok=True)

    # Frozen policy
    frozen_model_dir = Path(
        "robot_nav/models/MARL/marlTD3/checkpoint/Mar.02_obstacle_14robot_reward8"
    )
    frozen_model_name = "TD3-MARL-obstacle-14robots"

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

    # ---- Frozen policy ----
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
    for p in frozen_policy.actor.parameters():
        p.requires_grad = False
    for p in frozen_policy.critic.parameters():
        p.requires_grad = False
    frozen_policy.actor.eval()
    frozen_policy.critic.eval()
    print("Frozen policy loaded and frozen.")

    # ---- Mixing network ----
    mixing_net = MixingNetwork(
        embedding_dim=embedding_dim * 2,
        hidden_dim=128,
    ).to(device)
    mixing_optimizer = torch.optim.Adam(mixing_net.parameters(), lr=lr_mixing)
    print(f"MixingNetwork parameters: {sum(p.numel() for p in mixing_net.parameters()):,}")

    # ---- Group catalogue ----
    groups = generate_all_combinations(
        n=num_robots, min_size=min_group_size, max_size=max_group_size
    )
    print_combination_statistics(groups, n=num_robots, max_display_per_size=5)
    print(f"Using {len(groups)} groups (size {min_group_size}–{max_group_size})")

    # ---- EMA baselines (one per group size) ----
    baselines = {}  # group_size → running EMA of R_norm
    for sz in range(min_group_size, max_group_size + 1):
        baselines[sz] = 0.0

    # ---- Logging ----
    writer = SummaryWriter(comment="GroupMixing-REINFORCE-14robots")

    # ---- Initial reset ----
    (
        poses, distance, cos_val, sin_val, collision, goal, a, reward,
        positions, goal_positions, obstacle_states
    ) = sim.reset(random_obstacles=True)

    steps_since_reset = 0
    current_noise_std = logit_noise_std

    print(f"\nStarting REINFORCE training for {max_updates} updates …")
    print(f"  Batch size:       {batch_size}")
    print(f"  Rollout horizon:  {rollout_horizon}")
    print(f"  Env steps/update: {batch_size * rollout_horizon}")
    print(f"  Noise std:        {logit_noise_std} → {noise_std_min}")
    print("=" * 70)

    # ================================================================
    #  REINFORCE TRAINING LOOP
    # ================================================================
    for update in range(max_updates):

        batch_losses = []
        batch_returns = []
        batch_returns_norm = []
        batch_group_sizes = []
        batch_collisions = 0
        batch_goals = 0

        for rollout_idx in range(batch_size):

            # ---- Reset env if needed ----
            if (
                steps_since_reset >= max_steps_before_reset
                or any(collision)
                or outside_of_bounds(poses, sim)
            ):
                (
                    poses, distance, cos_val, sin_val, collision, goal, a, reward,
                    positions, goal_positions, obstacle_states
                ) = sim.reset(random_obstacles=True)
                steps_since_reset = 0

            # ---- 1. Sample random group G ----
            active_group = random.choice(groups)
            group_size = len(active_group)

            # ---- 2. Get embeddings from frozen GAT (no grad) ----
            robot_state, _ = frozen_policy.prepare_state(
                poses, distance, cos_val, sin_val, collision, a, goal_positions
            )
            robot_obs = np.array(robot_state)

            embeddings = get_embeddings_from_frozen_actor(
                frozen_policy.actor, robot_obs, obstacle_states, device
            )  # (N, 512), detached

            # ---- 3. Compute logits = f_lin(e_i) for i in G ----
            g_idx = torch.tensor(active_group, dtype=torch.long, device=device)
            g_emb = embeddings[g_idx]  # (G, 512)
            logits = mixing_net(g_emb)  # (G,)

            # ---- 4. Sample noise ONCE: ε ~ N(0, σ²) ----
            epsilon = torch.randn_like(logits) * current_noise_std

            # ---- 5. noisy_logits (detached target for log_prob) ----
            noisy_logits = (logits.detach() + epsilon)  # no grad through this

            # ---- 6. log_prob with grad through logits ----
            # log π = -0.5 * Σ ((noisy_logit_i - logit_i)² / σ²)
            # noisy_logits is detached, logits has grad → ∇_θ log π ≠ 0
            log_prob = -0.5 * ((noisy_logits - logits) ** 2).sum() / (current_noise_std ** 2)

            # ---- 7. α = softmax(noisy_logits) — fixed for the rollout ----
            # Apply √|G| scaling
            scaled_noisy = noisy_logits * math.sqrt(group_size) if group_size > 1 else noisy_logits
            alpha = F.softmax(scaled_noisy, dim=0).detach()  # (G,), detached

            # ---- 8. Execute rollout_horizon steps ----
            rollout_rewards = []
            group_set = set(active_group)

            for t in range(rollout_horizon):
                # Get current state
                robot_state_t, _ = frozen_policy.prepare_state(
                    poses, distance, cos_val, sin_val, collision, a, goal_positions
                )
                robot_obs_t = np.array(robot_state_t)

                # Frozen actor → fresh raw actions
                with torch.no_grad():
                    raw_action, combined_weights = frozen_policy.get_action(
                        robot_obs_t, obstacle_states, add_noise=False
                    )

                # Build coupled action using fixed α
                # Scale raw velocities: [-1,1] → [0,0.5]
                a_in = []
                scaled_vels = [(raw_action[idx][0] + 1) / 4 for idx in active_group]
                v_shared = sum(
                    float(alpha[li]) * scaled_vels[li]
                    for li in range(group_size)
                )

                # Angular velocity: couple if group > 3
                if group_size > 3:
                    w_shared = sum(
                        float(alpha[li]) * raw_action[active_group[li]][1]
                        for li in range(group_size)
                    )
                else:
                    w_shared = None

                for i in range(num_robots):
                    if i in group_set:
                        w = w_shared if w_shared is not None else float(raw_action[i][1])
                        a_in.append([v_shared, w])
                    else:
                        a_in.append([0.0, 0.0])

                # Step environment
                (
                    poses, distance, cos_val, sin_val, collision, goal, a, reward,
                    positions, goal_positions, obstacle_states
                ) = sim.step(a_in, None, combined_weights)

                steps_since_reset += 1

                # Collect reward for group members only, averaged over group
                group_reward = sum(reward[i] for i in active_group) / group_size
                rollout_rewards.append(group_reward)

                batch_goals += sum(goal)
                batch_collisions += sum(collision)

                # Early termination on collision or out-of-bounds
                if any(collision):
                    break

            # ---- 9. Discounted return R ----
            R = 0.0
            for r in reversed(rollout_rewards):
                R = r + gamma * R

            # ---- 10. R is already per-robot (averaged in step 8) ----
            R_norm = R

            # ---- 11. loss = -log_prob * (R_norm - baseline) ----
            advantage = R_norm - baselines[group_size]
            loss = -log_prob * advantage

            batch_losses.append(loss)
            batch_returns.append(R)
            batch_returns_norm.append(R_norm)
            batch_group_sizes.append(group_size)

        # ---- Aggregate batch loss and update ----
        total_loss = torch.stack(batch_losses).mean()

        mixing_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mixing_net.parameters(), max_grad_norm)
        mixing_optimizer.step()

        # ---- Update baselines per group size ----
        for sz in baselines:
            sz_returns = [
                batch_returns_norm[i]
                for i in range(batch_size)
                if batch_group_sizes[i] == sz
            ]
            if sz_returns:
                avg_ret = sum(sz_returns) / len(sz_returns)
                baselines[sz] = (
                    baseline_momentum * baselines[sz]
                    + (1 - baseline_momentum) * avg_ret
                )

        # ---- Decay noise ----
        current_noise_std = max(noise_std_min, current_noise_std * noise_decay)

        # ---- Logging ----
        avg_return = sum(batch_returns) / batch_size
        avg_return_norm = sum(batch_returns_norm) / batch_size

        writer.add_scalar("train/loss", total_loss.item(), update)
        writer.add_scalar("train/avg_return", avg_return, update)
        writer.add_scalar("train/avg_return_per_robot", avg_return_norm, update)
        writer.add_scalar("train/noise_std", current_noise_std, update)
        writer.add_scalar("train/batch_collisions", batch_collisions, update)
        writer.add_scalar("train/batch_goals", batch_goals, update)
        for sz in baselines:
            writer.add_scalar(f"train/baseline_size{sz}", baselines[sz], update)

        # ---- Console logging ----
        if update % 50 == 0:
            print(
                f"Update {update}/{max_updates} | "
                f"Loss: {total_loss.item():.4f} | "
                f"AvgReturn: {avg_return:.2f} | "
                f"AvgReturn/robot: {avg_return_norm:.2f} | "
                f"σ: {current_noise_std:.4f} | "
                f"b[2]: {baselines.get(2, 0):.2f}  b[3]: {baselines.get(3, 0):.2f} | "
                f"Coll: {batch_collisions}  Goals: {batch_goals}"
            )

        # ---- Checkpointing ----
        if (update + 1) % checkpoint_every == 0:
            ckpt_path = save_directory / f"mixing_net_update{update+1}.pth"
            torch.save(mixing_net.state_dict(), ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path}")

    # ---- Final save ----
    final_path = save_directory / "mixing_net_final.pth"
    torch.save(mixing_net.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
