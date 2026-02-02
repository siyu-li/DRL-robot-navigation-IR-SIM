"""
Test script for MARL with Obstacle Graph Nodes.

This script tests the obstacle-aware TD3 model in evaluation mode.
Tracks individual robot trajectories and calculates success rate.
"""

from pathlib import Path

import torch
import numpy as np
import logging
import time
import random
from shapely.geometry import Point

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


def robot_outside_bounds(pose, sim):
    """Check if a specific robot is outside world boundaries."""
    if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
        return True
    if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
        return True
    return False


def reset_single_robot(sim, robot_idx, init_states):
    """
    Reset a single robot to a new random position and goal.
    
    Args:
        sim: The simulation environment
        robot_idx: Index of the robot to reset
        init_states: List of current robot positions to avoid overlap
    
    Returns:
        Updated init_states list
    """
    x_min = sim.x_range[0] + 1
    x_max = sim.x_range[1] - 1
    y_min = sim.y_range[0] + 1
    y_max = sim.y_range[1] - 1
    
    robot = sim.env.robot_list[robot_idx]
    
    conflict = True
    while conflict:
        conflict = False
        robot_state = [
            [random.uniform(x_min, x_max)],
            [random.uniform(y_min, y_max)],
            [random.uniform(-np.pi, np.pi)],
        ]
        pos = [robot_state[0][0], robot_state[1][0]]
        
        # Check robot-robot spacing
        for j, loc in enumerate(init_states):
            if j == robot_idx:
                continue
            vector = [pos[0] - loc[0], pos[1] - loc[1]]
            if np.linalg.norm(vector) < 0.6:
                conflict = True
                break
        
        # Check robot-obstacle clearance
        if not conflict:
            robot_point = Point(pos[0], pos[1])
            for obs in sim.env.obstacle_list:
                if obs.geometry.distance(robot_point) < 0.5:
                    conflict = True
                    break
    
    # Update init_states
    init_states[robot_idx] = pos
    
    # Set robot state
    robot.set_state(state=np.array(robot_state), init=True)
    
    # Set new random goal
    robot.set_random_goal(
        obstacle_list=sim.env.obstacle_list,
        init=True,
        range_limits=[
            [sim.x_range[0] + 1, sim.y_range[0] + 1, -np.pi],
            [sim.x_range[1] - 1, sim.y_range[1] - 1, np.pi],
        ],
    )
    
    return init_states


def main(args=None):
    """Main test function."""

    # ---- Hyperparameters ----
    action_dim = 2
    max_action = 1
    state_dim = 11
    obstacle_state_dim = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_episodes = 3
    max_steps = 500  # Steps per episode
    render_delay = 0.05  # Delay between steps for visualization

    # ---- Instantiate environment ----
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle.yaml",
        disable_plotting=False,  # Enable plotting for visualization
        reward_phase=5,
        per_robot_goal_reset=False,  # We handle resets manually
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
        model_name="TD3-MARL-obstacle-6robots_epoch2400",
        load_model_name="TD3-MARL-obstacle-6robots_epoch2400",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots_v2"),
    )

    # ---- Statistics tracking ----
    # Per-trajectory tracking
    completed_trajectories = []  # List of (robot_idx, 'goal' or 'collision', cumulative_reward)
    trajectory_rewards = [0.0] * sim.num_robots  # Cumulative reward for current trajectory
    
    # Episode-level tracking
    total_steps = 0
    episode_total_rewards = []

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
        
        # Track current robot positions for collision-free respawn
        current_positions = [[p[0], p[1]] for p in poses]
        
        # Reset trajectory rewards for new episode
        trajectory_rewards = [0.0] * sim.num_robots

        while steps < max_steps:
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
            total_steps += 1
            
            # Update current positions
            current_positions = [[p[0], p[1]] for p in poses]

            # Print instant rewards for each robot
            reward_str = " | ".join([f"R{i}:{r:+.2f}" for i, r in enumerate(reward)])
            print(f"[Ep {episode+1} Step {steps}] {reward_str} | Total: {step_total_reward:+.2f}")

            # Accumulate rewards for trajectories
            for i in range(sim.num_robots):
                trajectory_rewards[i] += reward[i]

            # Check for completed trajectories and reset robots
            robots_to_reset = []
            for i in range(sim.num_robots):
                if collision[i]:
                    # Trajectory ended with collision
                    completed_trajectories.append((i, 'collision', trajectory_rewards[i]))
                    print(f"  >> Robot {i} COLLISION - Trajectory reward: {trajectory_rewards[i]:.2f}")
                    trajectory_rewards[i] = 0.0
                    robots_to_reset.append(i)
                elif goal[i]:
                    # Trajectory ended with goal reach
                    completed_trajectories.append((i, 'goal', trajectory_rewards[i]))
                    print(f"  >> Robot {i} GOAL REACHED - Trajectory reward: {trajectory_rewards[i]:.2f}")
                    trajectory_rewards[i] = 0.0
                    robots_to_reset.append(i)
                elif robot_outside_bounds(poses[i], sim):
                    # Trajectory ended by going out of bounds (treat as collision)
                    completed_trajectories.append((i, 'collision', trajectory_rewards[i]))
                    print(f"  >> Robot {i} OUT OF BOUNDS - Trajectory reward: {trajectory_rewards[i]:.2f}")
                    trajectory_rewards[i] = 0.0
                    robots_to_reset.append(i)

            # Reset robots that completed their trajectories
            for robot_idx in robots_to_reset:
                current_positions = reset_single_robot(sim, robot_idx, current_positions)

            # Visualization delay
            time.sleep(render_delay)

        episode_total_rewards.append(episode_reward)

        # Episode summary
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_total_rewards[-10:])
            
            # Calculate current success rate
            total_completed = len(completed_trajectories)
            successful = sum(1 for t in completed_trajectories if t[1] == 'goal')
            success_rate = (successful / total_completed * 100) if total_completed > 0 else 0
            
            print("=" * 80)
            print(
                f"Episode {episode + 1}/{max_episodes} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Trajectories: {total_completed} | "
                f"Success Rate: {success_rate:.1f}%"
            )
            print("=" * 80)

    # Final statistics
    total_completed = len(completed_trajectories)
    successful_trajectories = [t for t in completed_trajectories if t[1] == 'goal']
    collision_trajectories = [t for t in completed_trajectories if t[1] == 'collision']
    
    success_count = len(successful_trajectories)
    collision_count = len(collision_trajectories)
    success_rate = (success_count / total_completed * 100) if total_completed > 0 else 0
    
    # Compute average rewards for successful vs failed trajectories
    avg_success_reward = np.mean([t[2] for t in successful_trajectories]) if successful_trajectories else 0
    avg_collision_reward = np.mean([t[2] for t in collision_trajectories]) if collision_trajectories else 0

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total episodes: {max_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Total completed trajectories: {total_completed}")
    print(f"  - Successful (goal reached): {success_count}")
    print(f"  - Failed (collision/out-of-bounds): {collision_count}")
    print(f"\nSUCCESS RATE: {success_rate:.2f}%")
    print(f"\nAverage trajectory reward (successful): {avg_success_reward:.2f}")
    print(f"Average trajectory reward (failed): {avg_collision_reward:.2f}")
    print(f"Average episode reward: {np.mean(episode_total_rewards):.2f}")
    
    # Per-robot statistics
    print("\n" + "-" * 40)
    print("Per-Robot Statistics:")
    for robot_idx in range(sim.num_robots):
        robot_trajectories = [t for t in completed_trajectories if t[0] == robot_idx]
        robot_success = sum(1 for t in robot_trajectories if t[1] == 'goal')
        robot_total = len(robot_trajectories)
        robot_rate = (robot_success / robot_total * 100) if robot_total > 0 else 0
        print(f"  Robot {robot_idx}: {robot_success}/{robot_total} successful ({robot_rate:.1f}%)")


if __name__ == "__main__":
    main()
