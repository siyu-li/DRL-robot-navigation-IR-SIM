"""
Training script for MARL TD3 with Obstacle Graph Nodes.

This script trains a multi-agent TD3 policy with graph attention that includes
obstacle nodes. Obstacles are static graph nodes that send messages to robots
but do not have actions.

Usage:
    python -m robot_nav.marl_train_obstacle

Key features:
- Obstacle nodes in the graph attention network
- Robot-obstacle clearance computed via Shapely geometry
- Separate BCE supervision for robot-robot and robot-obstacle edges
- Obstacle proximity penalty in reward function
"""

from pathlib import Path

import torch
import numpy as np
import logging

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.replay_buffer_obstacle import ReplayBufferObstacle

# Suppress IRSim warnings
logging.getLogger('irsim').setLevel(logging.ERROR)


def outside_of_bounds(poses, sim):
    """
    Check if any robot is outside the defined world boundaries.

    Args:
        poses (list): List of [x, y, theta] poses for each robot.
        sim: Simulation environment with x_range and y_range.

    Returns:
        bool: True if any robot is outside world boundaries.
    """
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def main(args=None):
    """Main training function for obstacle-aware MARL."""

    # ---- Hyperparameters ----
    action_dim = 2
    max_action = 1
    state_dim = 11  # Robot state dimension
    obstacle_state_dim = 4  # Obstacle state: [x, y, cos_h, sin_h]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_epochs = 2000
    epoch = 1
    episode = 0
    train_every_n = 10
    training_iterations = 80
    batch_size = 16
    max_steps = 300
    steps = 0
    save_every = 5
    buffer_size = 50000

    # Environment hyperparameters
    per_robot_goal_reset = True
    obstacle_proximity_threshold = 2.0  # For reward penalty

    # ---- Instantiate environment ----
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_lidar.yaml",
        disable_plotting=True,
        reward_phase=3,
        per_robot_goal_reset=per_robot_goal_reset,
        obstacle_proximity_threshold=obstacle_proximity_threshold,
    )

    print(f"Environment initialized:")
    print(f"  - Number of robots: {sim.num_robots}")
    print(f"  - Number of obstacles: {sim.num_obstacles}")
    print(f"  - World bounds: x={sim.x_range}, y={sim.y_range}")

    # ---- Instantiate model ----
    model = TD3Obstacle(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        obstacle_state_dim=obstacle_state_dim,
        device=device,
        save_every=save_every,
        load_model=False,
        model_name="TD3-MARL-obstacle",
        save_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/obstacle"),
    )

    # ---- Setup replay buffer ----
    replay_buffer = ReplayBufferObstacle(buffer_size=buffer_size)

    # ---- Take initial step in environment ----
    (
        poses, distance, cos, sin, collision, goal, a, reward,
        positions, goal_positions, obstacle_states
    ) = sim.step([[0, 0] for _ in range(sim.num_robots)], None)

    running_goals = 0
    running_collisions = 0
    running_timesteps = 0

    print(f"\nStarting training...")
    print(f"Initial obstacle states shape: {obstacle_states.shape}")

    # ---- Main training loop ----
    while epoch < max_epochs:
        # Prepare robot state
        robot_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )

        # Get action from model
        action, combined_weights = model.get_action(
            np.array(robot_state), obstacle_states, add_noise=True
        )

        # Scale action for environment
        a_in = [[(act[0] + 1) / 4, act[1]] for act in action]

        # Step environment
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, next_obstacle_states
        ) = sim.step(a_in, None, combined_weights)

        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_timesteps += 1

        # Prepare next state
        next_robot_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )

        # Add to replay buffer
        replay_buffer.add(
            robot_state,
            obstacle_states,
            action,
            reward,
            terminal,
            next_robot_state,
            next_obstacle_states,
        )

        # Update obstacle states for next iteration
        obstacle_states = next_obstacle_states

        steps += 1
        episode += 1

        # Check termination conditions
        if (
            any(collision)
            or all(goal)
            or steps >= max_steps
            or outside_of_bounds(poses, sim)
        ):
            (
                poses, distance, cos, sin, collision, goal, a, reward,
                positions, goal_positions, obstacle_states
            ) = sim.reset()

            steps = 0
            epoch += 1

            # Training
            if episode >= train_every_n and replay_buffer.size() >= batch_size:
                model.train(
                    replay_buffer,
                    training_iterations,
                    batch_size,
                    connection_proximity_threshold_rr=4.0,
                    connection_proximity_threshold_ro=2.0,
                )
                episode = 0

                # Logging
                if epoch % 10 == 0:
                    avg_goals = running_goals / max(running_timesteps, 1) * 100
                    avg_collisions = running_collisions / max(running_timesteps, 1) * 100
                    print(
                        f"Epoch {epoch}/{max_epochs} | "
                        f"Buffer: {replay_buffer.size()} | "
                        f"Goals: {avg_goals:.1f}% | "
                        f"Collisions: {avg_collisions:.1f}%"
                    )
                    running_goals = 0
                    running_collisions = 0
                    running_timesteps = 0

    print("\nTraining complete!")
    model.save(filename=model.model_name, directory=model.save_directory)


if __name__ == "__main__":
    main()
