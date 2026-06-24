"""
Test script for MARL with Obstacle Graph Nodes (14 Robots).

This script tests the obstacle-aware TD3 model with 14 robots in evaluation mode.
All robots are reset together when all reach goals or any collision occurs.
"""

from pathlib import Path

import torch
import numpy as np
import logging
import time

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

# Suppress IRSim warnings - irsim uses loguru, not standard logging
from loguru import logger
logger.disable("irsim")


def main(args=None):
    """Main test function."""

    # ---- Hyperparameters ----
    action_dim = 2
    max_action = 1
    state_dim = 11
    obstacle_state_dim = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_episodes = 100
    max_steps = 500  # Steps per episode
    render_delay = 0.05  # Delay between steps for visualization

    # ---- Instantiate environment ----
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
        disable_plotting=False,  # Enable plotting for visualization
        reward_phase=6,
        per_robot_goal_reset=True,  # Enable dwell logic for correct post-goal rewards
        obstacle_proximity_threshold=1.5,
        goal_dwell_min=999999,        # Never respawn during test — dwell forever
        goal_respawn_prob=1.0,
        station_keeping_reward=0.5,
    )

    print(f"Environment initialized:")
    print(f"  - Number of robots: {sim.num_robots}")
    print(f"  - Number of obstacles: {sim.num_obstacles}")

    # ---- Instantiate model ----
    model = TD3Obstacle(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        obstacle_state_dim=obstacle_state_dim,
        device=device,
        load_model=True,
        model_name="TD3-MARL-obstacle-14robots",
        load_model_name="TD3-MARL-obstacle-14robots",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/Mar.15_obstacle_14robot"),

        # model_name="TD3-MARL-obstacle-14robots-gpu",
        # load_model_name="TD3-MARL-obstacle-14robots-gpu_epoch800",
        # load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/Feb.10_obstacle_14robot_transfer_gpu"),
    )

    # ---- Statistics tracking ----
    episode_rewards = []
    episode_steps = []
    episode_outcomes = []  # 'all_goals', 'collision', 'timeout'
    successful_episodes = 0
    collision_episodes = 0
    timeout_episodes = 0

    print(f"\nStarting evaluation for {max_episodes} episodes...")
    print(f"Max steps per episode: {max_steps}")
    print(f"Number of robots: {sim.num_robots}")
    print("-" * 80)

    for episode in range(max_episodes):
        # Reset environment (with randomized obstacles)
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, obstacle_states
        ) = sim.reset(random_obstacles=True)

        episode_reward = 0
        steps = 0
        episode_done = False
        outcome = None

        while steps < max_steps and not episode_done:
            # Prepare state
            robot_state, _ = model.prepare_state(
                poses, distance, cos, sin, collision, a, goal_positions
            )

            # Get action (no noise during evaluation)
            action, combined_weights = model.get_action(
                np.array(robot_state), obstacle_states, add_noise=False
            )

            # Scale action
            a_in = [[(act[0] + 1) / 4, act[1]] for act in action]

            # Step environment
            (
                poses, distance, cos, sin, collision, goal, a, reward,
                positions, goal_positions, next_obstacle_states
            ) = sim.step(a_in, None, combined_weights)

            obstacle_states = next_obstacle_states
            step_total_reward = sum(reward)
            episode_reward += step_total_reward
            steps += 1

            # Print step information periodically
            if steps % 50 == 0:
                reward_str = " | ".join([f"R{i}:{r:+.2f}" for i, r in enumerate(reward)])
                goals_reached = sum(goal)
                num_dwelling = sum(1 for c in sim.dwell_counters if c >= 0)
                print(f"[Ep {episode+1} Step {steps}] {reward_str} | Total: {step_total_reward:+.2f} | Goals: {goals_reached}/{sim.num_robots} | Dwelling: {num_dwelling}")

            # Check episode termination conditions
            if any(collision):
                # Any collision ends the episode
                episode_done = True
                outcome = 'collision'
                collision_episodes += 1
                print(f"  >> Episode {episode+1} ended: COLLISION at step {steps}")
            elif all(goal):
                # All robots reached their goals
                episode_done = True
                outcome = 'all_goals'
                successful_episodes += 1
                print(f"  >> Episode {episode+1} ended: ALL GOALS REACHED at step {steps}")

            # Visualization delay
            time.sleep(render_delay)

        # Handle timeout
        if not episode_done:
            outcome = 'timeout'
            timeout_episodes += 1
            goals_reached = sum(goal)
            print(f"  >> Episode {episode+1} ended: TIMEOUT at step {steps} | Goals: {goals_reached}/{sim.num_robots}")

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_outcomes.append(outcome)

        # Episode summary
        print(f"Episode {episode+1}/{max_episodes} | Reward: {episode_reward:.1f} | Steps: {steps} | Outcome: {outcome}")
        
        # Periodic statistics
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            recent_success = sum(1 for o in episode_outcomes[-10:] if o == 'all_goals')
            
            print("=" * 80)
            print(f"Episodes {episode-8}-{episode+1} Summary:")
            print(f"  Avg Reward: {avg_reward:.1f}")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Success Rate (last 10): {recent_success}/10 ({recent_success*10}%)")
            print(f"  Overall Success Rate: {successful_episodes}/{episode+1} ({successful_episodes/(episode+1)*100:.1f}%)")
            print("=" * 80)

    # Final statistics
    total_episodes = len(episode_rewards)
    success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0
    collision_rate = (collision_episodes / total_episodes * 100) if total_episodes > 0 else 0
    timeout_rate = (timeout_episodes / total_episodes * 100) if total_episodes > 0 else 0
    
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    # Calculate average rewards by outcome
    success_rewards = [episode_rewards[i] for i, o in enumerate(episode_outcomes) if o == 'all_goals']
    collision_rewards = [episode_rewards[i] for i, o in enumerate(episode_outcomes) if o == 'collision']
    timeout_rewards = [episode_rewards[i] for i, o in enumerate(episode_outcomes) if o == 'timeout']
    
    avg_success_reward = np.mean(success_rewards) if success_rewards else 0
    avg_collision_reward = np.mean(collision_rewards) if collision_rewards else 0
    avg_timeout_reward = np.mean(timeout_rewards) if timeout_rewards else 0
    
    # Calculate average steps by outcome
    success_steps = [episode_steps[i] for i, o in enumerate(episode_outcomes) if o == 'all_goals']
    collision_steps = [episode_steps[i] for i, o in enumerate(episode_outcomes) if o == 'collision']
    
    avg_success_steps = np.mean(success_steps) if success_steps else 0
    avg_collision_steps = np.mean(collision_steps) if collision_steps else 0

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total episodes: {total_episodes}")
    print(f"Number of robots: {sim.num_robots}")
    print(f"\nEpisode Outcomes:")
    print(f"  - Successful (all goals): {successful_episodes} ({success_rate:.1f}%)")
    print(f"  - Collision: {collision_episodes} ({collision_rate:.1f}%)")
    print(f"  - Timeout: {timeout_episodes} ({timeout_rate:.1f}%)")
    print(f"\nOverall Statistics:")
    print(f"  - Average episode reward: {avg_reward:.2f}")
    print(f"  - Average episode steps: {avg_steps:.1f}")
    print(f"\nReward by Outcome:")
    print(f"  - Successful episodes: {avg_success_reward:.2f}")
    print(f"  - Collision episodes: {avg_collision_reward:.2f}")
    print(f"  - Timeout episodes: {avg_timeout_reward:.2f}")
    print(f"\nSteps by Outcome:")
    print(f"  - Successful episodes: {avg_success_steps:.1f}")
    print(f"  - Collision episodes: {avg_collision_steps:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
