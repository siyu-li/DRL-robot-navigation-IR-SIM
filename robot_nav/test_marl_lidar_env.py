"""
Test script for MARL_LIDAR_SIM environment.

This script tests:
- Environment initialization with 5 robots
- Random action execution
- LiDAR scan extraction and statistics
- Reset functionality (robot positions, goals, obstacles)
- Rendering visualization
"""

import numpy as np
import torch
import time


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def print_robot_states(poses, distances, goals, collisions, goal_positions):
    """Print state information for all robots."""
    print("\n  Robot States:")
    print("  " + "-" * 56)
    print(f"  {'Robot':<8} {'Position (x, y)':<20} {'θ':<10} {'Dist to Goal':<12} {'Goal':<6} {'Collision'}")
    print("  " + "-" * 56)
    
    for i, (pose, dist, goal, collision, goal_pos) in enumerate(
        zip(poses, distances, goals, collisions, goal_positions)
    ):
        pos_str = f"({pose[0]:6.2f}, {pose[1]:6.2f})"
        theta_str = f"{pose[2]:6.2f}"
        goal_str = "✓" if goal else ""
        coll_str = "⚠️" if collision else ""
        print(f"  {i:<8} {pos_str:<20} {theta_str:<10} {dist:<12.2f} {goal_str:<6} {coll_str}")
    
    print(f"\n  Goal Positions:")
    for i, goal_pos in enumerate(goal_positions):
        print(f"    Robot {i}: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")


def print_lidar_stats(lidar_scans, lidar_range_max):
    """Print LiDAR statistics for all robots."""
    print("\n  LiDAR Statistics (normalized values, multiply by {:.1f} for meters):".format(lidar_range_max))
    print("  " + "-" * 50)
    print(f"  {'Robot':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Num Beams'}")
    print("  " + "-" * 50)
    
    for i, scan in enumerate(lidar_scans):
        print(f"  {i:<8} {scan.min():<12.3f} {scan.max():<12.3f} {scan.mean():<12.3f} {len(scan)}")
    
    # Overall statistics
    all_scans = np.concatenate(lidar_scans)
    print("  " + "-" * 50)
    print(f"  {'Overall':<8} {all_scans.min():<12.3f} {all_scans.max():<12.3f} {all_scans.mean():<12.3f}")
    
    # Convert to actual distances
    print(f"\n  Actual distances (meters):")
    print(f"    Min: {all_scans.min() * lidar_range_max:.2f}m")
    print(f"    Max: {all_scans.max() * lidar_range_max:.2f}m")
    print(f"    Mean: {all_scans.mean() * lidar_range_max:.2f}m")


def generate_random_actions(num_robots, linear_range=(0.0, 0.5), angular_range=(-1.0, 1.0)):
    """Generate random actions for all robots."""
    actions = []
    for _ in range(num_robots):
        linear_vel = np.random.uniform(*linear_range)
        angular_vel = np.random.uniform(*angular_range)
        actions.append([linear_vel, angular_vel])
    return actions


def run_test():
    """Main test function."""
    from robot_nav.SIM_ENV.marl_lidar_sim import MARL_LIDAR_SIM
    
    # Configuration
    NUM_EPISODES = 3
    STEPS_PER_EPISODE = 100
    PRINT_INTERVAL = 20  # Print stats every N steps
    
    print_separator("MARL LiDAR Environment Test")
    print("\nConfiguration:")
    print(f"  - Number of episodes: {NUM_EPISODES}")
    print(f"  - Steps per episode: {STEPS_PER_EPISODE}")
    print(f"  - Print interval: {PRINT_INTERVAL} steps")
    
    # Initialize environment
    print("\nInitializing environment...")
    sim = MARL_LIDAR_SIM(
        world_file="robot_nav/worlds/multi_robot_world_lidar.yaml",
        disable_plotting=False,  # Enable rendering
        reward_phase=1,
        use_lidar=True,
        lidar_num_beams=180,
        lidar_range_max=7.0,
        random_obstacles=True,
        num_obstacles=5,
    )
    
    print(f"\nEnvironment initialized:")
    print(f"  - Number of robots: {sim.num_robots}")
    print(f"  - World X range: {sim.x_range}")
    print(f"  - World Y range: {sim.y_range}")
    print(f"  - LiDAR enabled: {sim.use_lidar}")
    print(f"  - LiDAR beams: {sim.lidar_num_beams}")
    print(f"  - LiDAR max range: {sim.lidar_range_max}m")
    print(f"  - Random obstacles: {sim.random_obstacles_enabled}")
    
    # Run episodes
    for episode in range(NUM_EPISODES):
        print_separator(f"Episode {episode + 1}/{NUM_EPISODES}")
        
        # Reset environment
        print("\nResetting environment...")
        result = sim.reset(random_obstacles=True)
        
        # Unpack reset result
        (
            poses, distances, coss, sins, collisions, goals,
            action, rewards, positions, goal_positions, lidar_scans
        ) = result
        
        print("\nInitial state after reset:")
        print_robot_states(poses, distances, goals, collisions, goal_positions)
        print_lidar_stats(lidar_scans, sim.lidar_range_max)
        
        # Episode statistics
        episode_rewards = [[] for _ in range(sim.num_robots)]
        episode_collisions = [0 for _ in range(sim.num_robots)]
        episode_goals_reached = [0 for _ in range(sim.num_robots)]
        
        # Run steps
        for step in range(STEPS_PER_EPISODE):
            # Generate random actions
            actions = generate_random_actions(sim.num_robots)
            
            # Create dummy connection tensor
            connection = torch.zeros((sim.num_robots, sim.num_robots - 1))
            
            # Step environment
            result = sim.step(actions, connection)
            
            # Unpack step result
            (
                poses, distances, coss, sins, collisions, goals,
                action, rewards, positions, goal_positions, lidar_scans
            ) = result
            
            # Track statistics
            for i in range(sim.num_robots):
                episode_rewards[i].append(rewards[i])
                if collisions[i]:
                    episode_collisions[i] += 1
                if goals[i]:
                    episode_goals_reached[i] += 1
            
            # Print periodic updates
            if (step + 1) % PRINT_INTERVAL == 0:
                print(f"\n  --- Step {step + 1}/{STEPS_PER_EPISODE} ---")
                print_robot_states(poses, distances, goals, collisions, goal_positions)
                print_lidar_stats(lidar_scans, sim.lidar_range_max)
                print(f"\n  Rewards this step: {[f'{r:.2f}' for r in rewards]}")
            
            # Small delay for visualization
            time.sleep(0.05)
        
        # Episode summary
        print_separator(f"Episode {episode + 1} Summary")
        print("\n  Per-Robot Statistics:")
        print("  " + "-" * 60)
        print(f"  {'Robot':<8} {'Total Reward':<15} {'Avg Reward':<15} {'Collisions':<12} {'Goals'}")
        print("  " + "-" * 60)
        
        for i in range(sim.num_robots):
            total_reward = sum(episode_rewards[i])
            avg_reward = np.mean(episode_rewards[i])
            print(f"  {i:<8} {total_reward:<15.2f} {avg_reward:<15.2f} {episode_collisions[i]:<12} {episode_goals_reached[i]}")
        
        print("\n  Press Enter to continue to next episode (or Ctrl+C to exit)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
            break
    
    print_separator("Test Complete")
    print("\nAll episodes finished.")
    
    # Print debug info
    debug_info = sim.get_debug_info()
    print("\nFinal Debug Info:")
    for key, value in debug_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run_test()
