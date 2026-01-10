"""
Data Collection Script for Coupled Action Policy Supervised Training.

This script runs the trained decentralized policy and saves rollout data
to YAML files for use in supervised training of the v_head.

The saved data format is compatible with SupervisedDatasetGenerator.

Usage:
    python -m robot_nav.coupled_action_data_collection \
        --model_name TDR-MARL-train \
        --num_episodes 100 \
        --output_path robot_nav/assets/coupled_action_data.yml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.marlTD3 import TD3
from robot_nav.SIM_ENV.marl_sim import MARL_SIM
from robot_nav.utils import MARLDataSaver


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect rollout data from decentralized policy"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="TDR-MARL-train",
        help="Name of trained decentralized model"
    )
    parser.add_argument(
        "--model_directory",
        type=str,
        default="robot_nav/models/MARL/marlTD3/checkpoint",
        help="Directory containing model weights"
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="igs",
        choices=["igs", "g2anet"],
        help="Attention mechanism type"
    )
    
    # Data collection configuration
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument(
        "--output_path",
        type=str,
        default="robot_nav/assets/coupled_action_data.yml",
        help="Output path for collected data"
    )
    
    # Environment configuration
    parser.add_argument("--num_robots", type=int, default=5, help="Number of robots")
    parser.add_argument("--state_dim", type=int, default=11, help="Per-robot state dimension")
    parser.add_argument("--action_dim", type=int, default=2, help="Per-robot action dimension")
    parser.add_argument(
        "--world_file",
        type=str,
        default="robot_nav/worlds/multi_robot_world.yaml",
        help="Path to world configuration file"
    )
    parser.add_argument("--disable_plotting", action="store_true", help="Disable visualization")
    
    # Noise configuration
    parser.add_argument("--add_noise", action="store_true", help="Add exploration noise to actions")
    parser.add_argument("--noise_std", type=float, default=0.2, help="Standard deviation of action noise")
    
    return parser.parse_args()


def outside_of_bounds(poses, center=(6, 6), half_size=10.5):
    """Check if any robot is outside world boundaries."""
    for pose in poses:
        norm_x = pose[0] - center[0]
        norm_y = pose[1] - center[1]
        if abs(norm_x) > half_size or abs(norm_y) > half_size:
            return True
    return False


def collect_data(
    model: TD3,
    sim: MARL_SIM,
    data_saver: MARLDataSaver,
    num_episodes: int = 100,
    max_steps: int = 600,
    add_noise: bool = False,
    verbose: bool = True
):
    """
    Collect rollout data from decentralized policy.
    
    Args:
        model: Trained TD3 model.
        sim: MARL simulation environment.
        data_saver: MARLDataSaver for storing data.
        num_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.
        add_noise: Whether to add exploration noise.
        verbose: Print progress.
    """
    # Initialize connections tensor
    connections = torch.tensor(
        [[0.0 for _ in range(sim.num_robots - 1)] for _ in range(sim.num_robots)]
    )
    
    total_samples = 0
    total_goals = 0
    total_collisions = 0
    
    pbar = tqdm(range(num_episodes), desc="Collecting data") if verbose else range(num_episodes)
    
    for episode in pbar:
        # Reset environment
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions
        ) = sim.reset()
        
        steps = 0
        
        while steps < max_steps:
            # Prepare state
            state, terminal = model.prepare_state(
                poses, distance, cos, sin, collision, a, goal_positions
            )
            
            # Get action from model
            action, connection, combined_weights = model.get_action(
                np.array(state), add_noise=add_noise
            )
            
            # Convert to a_in format for simulation
            a_in = [[(a[0] + 1) / 4, a[1]] for a in action]
            
            # Step simulation
            (
                poses, distance, cos, sin, collision, goal, a, reward,
                positions, goal_positions
            ) = sim.step(a_in, connection, combined_weights)
            
            # Save data point
            data_saver.add(
                poses=poses,
                distances=distance,
                cos_vals=cos,
                sin_vals=sin,
                collisions=collision,
                goals=goal,
                actions=action.tolist(),  # Raw action in [-1, 1]
                goal_positions=goal_positions
            )
            
            total_samples += 1
            total_goals += sum(goal)
            total_collisions += sum(collision)
            
            steps += 1
            
            # Check termination
            outside = outside_of_bounds(poses)
            if sum(collision) > 0.5 or outside or all(goal):
                break
        
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({
                "samples": total_samples,
                "goals": total_goals,
                "cols": total_collisions
            })
    
    return total_samples, total_goals, total_collisions


def main():
    """Main data collection function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create simulation environment
    logger.info("Creating simulation environment...")
    sim = MARL_SIM(
        world_file=args.world_file,
        disable_plotting=args.disable_plotting,
        reward_phase=2
    )
    
    # Verify robot count
    if sim.num_robots != args.num_robots:
        logger.warning(
            f"Simulation has {sim.num_robots} robots but expected {args.num_robots}. "
            f"Using {sim.num_robots}."
        )
        args.num_robots = sim.num_robots
    
    # Create model
    logger.info("Loading decentralized policy...")
    model = TD3(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        max_action=1,
        num_robots=args.num_robots,
        device=device,
        save_every=0,
        load_model=True,
        model_name="data_collection",
        load_model_name=args.model_name,
        load_directory=Path(args.model_directory),
        attention=args.attention
    )
    
    # Create data saver
    data_saver = MARLDataSaver(filepath=args.output_path)
    
    # Collect data
    logger.info(f"Collecting data for {args.num_episodes} episodes...")
    total_samples, total_goals, total_collisions = collect_data(
        model=model,
        sim=sim,
        data_saver=data_saver,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        add_noise=args.add_noise,
        verbose=True
    )
    
    # Save data
    data_saver.save()
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Episodes collected: {args.num_episodes}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total goals: {total_goals}")
    logger.info(f"Total collisions: {total_collisions}")
    logger.info(f"Data saved to: {args.output_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
