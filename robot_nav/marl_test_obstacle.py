"""
Test script for MARL with Obstacle Graph Nodes.

This script tests the obstacle-aware TD3 model in evaluation mode.
"""

from pathlib import Path

import torch
import numpy as np
import logging
import time

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

# Suppress IRSim warnings
logging.getLogger('irsim').setLevel(logging.ERROR)


def outside_of_bounds(poses, sim):
    """Check if any robot is outside world boundaries."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


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
    max_steps = 300
    render_delay = 0.05  # Delay between steps for visualization

    # ---- Instantiate environment ----
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_lidar.yaml",
        disable_plotting=False,  # Enable plotting for visualization
        reward_phase=1,
        per_robot_goal_reset=True,
        obstacle_proximity_threshold=1.5,
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
        model_name="TD3-MARL-obstacle",
        load_model_name="TD3-MARL-obstacle",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/obstacle"),
    )

    # ---- Statistics tracking ----
    total_goals = 0
    total_collisions = 0
    total_steps = 0
    episode_rewards = []

    print(f"\nStarting evaluation for {max_episodes} episodes...")

    for episode in range(max_episodes):
        # Reset environment
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, obstacle_states
        ) = sim.reset()

        episode_reward = 0
        steps = 0

        while True:
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
            episode_reward += sum(reward)
            steps += 1
            total_steps += 1

            # Track goals and collisions
            total_goals += sum(goal)
            total_collisions += sum(collision)

            # Visualization delay
            time.sleep(render_delay)

            # Check termination
            if (
                any(collision)
                or all(goal)
                or steps >= max_steps
                or outside_of_bounds(poses, sim)
            ):
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode + 1}/{max_episodes} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Goals: {total_goals} | "
                f"Collisions: {total_collisions}"
            )

    # Final statistics
    print("\n" + "=" * 50)
    print("Evaluation Complete")
    print("=" * 50)
    print(f"Total episodes: {max_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Total goals reached: {total_goals}")
    print(f"Total collisions: {total_collisions}")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Goal rate: {total_goals / total_steps * 100:.2f}%")
    print(f"Collision rate: {total_collisions / total_steps * 100:.2f}%")


if __name__ == "__main__":
    main()
