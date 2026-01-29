"""
Evaluation Script for Group-based Coupled Action Policy with Obstacles.

This script evaluates the group-based coupled action policy in simulation by:
- Randomly selecting a group of 2 or 3 robots every N steps
- Using v_shared predicted by the trained v_head (pooled from group embeddings)
- Using w_i from the frozen omega head

Reports:
- Collision rate
- Success rate / goal reached
- Average time-to-goal / progress per step
- Group selection statistics

Usage:
    python -m robot_nav.coupled_action_group_eval
"""

import logging
import random
import statistics
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.coupled_action_policy_obstacle import (
    CoupledActionPolicyObstacle,
)
from robot_nav.models.MARL.marlTD3.supervised_dataset_group import generate_all_groups
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE


# ============================================================================
# Configuration Dictionary - Edit these values directly
# ============================================================================
CONFIG = {
    # Model configuration
    "model_name": "coupled_action_group_obstacle_best",
    "model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/group_policy",
    "pretrained_model_name": "TD3-MARL-obstacle-6robots",
    "pretrained_directory": "robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots",
    
    # Evaluation configuration
    "test_scenarios": 20,  # Number of test episodes
    "max_steps": 500,  # Max steps per episode
    "disable_plotting": False,  # Set to True to disable visualization
    
    # Group selection configuration
    "group_switch_interval": 10,  # Switch group every N steps
    "group_sizes": [2, 3],  # Possible group sizes to randomly select
    "use_all_combinations": True,  # If True, select from all combinations; if False, random sampling
    
    # Policy configuration
    "num_robots": 6,
    "num_obstacles": 4,
    "state_dim": 11,
    "obstacle_state_dim": 4,
    "embedding_dim": 256,
    "v_min": 0.0,
    "v_max": 0.5,
    "pooling": "mean",
    
    # World configuration
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle.yaml",
    "obstacle_proximity_threshold": 1.5,
}


def outside_of_bounds(poses: List[List[float]], sim) -> bool:
    """
    Check if any robot is outside the defined world boundaries.
    
    Args:
        poses: List of [x, y, theta] poses for each robot.
        sim: Simulation environment with x_range and y_range.
        
    Returns:
        True if any robot is outside bounds.
    """
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def select_random_group(
    num_robots: int,
    group_sizes: List[int],
    all_groups: Optional[List[List[int]]] = None
) -> List[int]:
    """
    Randomly select a group of robots.
    
    Args:
        num_robots: Total number of robots.
        group_sizes: List of possible group sizes.
        all_groups: Pre-computed list of all group combinations.
            If provided, samples from this list.
            
    Returns:
        List of robot indices in the selected group.
    """
    if all_groups is not None:
        return random.choice(all_groups)
    else:
        # Random sampling
        group_size = random.choice(group_sizes)
        return sorted(random.sample(range(num_robots), group_size))


class GroupEvaluationMetrics:
    """Track evaluation metrics across episodes with group statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.episode_rewards = []
        self.episode_goals = []
        self.episode_collisions = []
        self.episode_steps = []
        self.v_shared_values = []
        self.progress_per_step = []
        self.group_selections = []  # Track which groups were selected
        self.group_size_counts = {2: 0, 3: 0}  # Count by group size
    
    def add_episode(
        self,
        total_reward: float,
        goals_reached: int,
        collisions: int,
        steps: int,
        v_shared_history: List[float],
        progress: float,
        groups_used: List[List[int]]
    ):
        """Add metrics for a completed episode."""
        self.episode_rewards.append(total_reward)
        self.episode_goals.append(goals_reached)
        self.episode_collisions.append(collisions)
        self.episode_steps.append(steps)
        self.v_shared_values.extend(v_shared_history)
        self.progress_per_step.append(progress / max(steps, 1))
        self.group_selections.extend(groups_used)
        
        # Count group sizes
        for group in groups_used:
            size = len(group)
            if size in self.group_size_counts:
                self.group_size_counts[size] += 1
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        total_groups = sum(self.group_size_counts.values())
        
        return {
            "num_episodes": len(self.episode_rewards),
            "avg_reward": statistics.mean(self.episode_rewards),
            "std_reward": statistics.stdev(self.episode_rewards) if len(self.episode_rewards) > 1 else 0,
            "avg_goals": statistics.mean(self.episode_goals),
            "std_goals": statistics.stdev(self.episode_goals) if len(self.episode_goals) > 1 else 0,
            "avg_collisions": statistics.mean(self.episode_collisions),
            "std_collisions": statistics.stdev(self.episode_collisions) if len(self.episode_collisions) > 1 else 0,
            "collision_rate": sum(1 for c in self.episode_collisions if c > 0) / len(self.episode_collisions),
            "success_rate": sum(1 for g, c in zip(self.episode_goals, self.episode_collisions) 
                               if g > 0 and c == 0) / len(self.episode_goals),
            "avg_steps": statistics.mean(self.episode_steps),
            "avg_v_shared": statistics.mean(self.v_shared_values) if self.v_shared_values else 0,
            "std_v_shared": statistics.stdev(self.v_shared_values) if len(self.v_shared_values) > 1 else 0,
            "avg_progress_per_step": statistics.mean(self.progress_per_step) if self.progress_per_step else 0,
            "total_group_switches": total_groups,
            "group_size_2_ratio": self.group_size_counts[2] / total_groups if total_groups > 0 else 0,
            "group_size_3_ratio": self.group_size_counts[3] / total_groups if total_groups > 0 else 0,
        }


def evaluate_group_policy(
    policy: CoupledActionPolicyObstacle,
    sim: MARL_SIM_OBSTACLE,
    num_episodes: int = 100,
    max_steps: int = 500,
    group_switch_interval: int = 10,
    group_sizes: List[int] = [2, 3],
    use_all_combinations: bool = True,
    verbose: bool = True
) -> GroupEvaluationMetrics:
    """
    Evaluate the group-based coupled action policy in simulation.
    
    Args:
        policy: Trained CoupledActionPolicyObstacle.
        sim: MARL simulation environment with obstacles.
        num_episodes: Number of test episodes.
        max_steps: Maximum steps per episode.
        group_switch_interval: Switch active group every N steps.
        group_sizes: Possible group sizes for random selection.
        use_all_combinations: If True, select from all pre-computed combinations.
        verbose: Print progress.
        
    Returns:
        GroupEvaluationMetrics with recorded results.
    """
    metrics = GroupEvaluationMetrics()
    
    # Pre-compute all group combinations if requested
    if use_all_combinations:
        all_groups = generate_all_groups(sim.num_robots, group_sizes)
        print(f"Using {len(all_groups)} pre-computed group combinations")
    else:
        all_groups = None
    
    pbar = tqdm(range(num_episodes), desc="Evaluating") if verbose else range(num_episodes)
    
    for episode in pbar:
        # Reset environment
        (
            poses, distance, cos, sin, collision, goal, a, reward, 
            positions, goal_positions, obstacle_states
        ) = sim.reset(random_obstacles=True)
        
        episode_reward = 0.0
        episode_goals = 0
        episode_collisions = 0
        episode_v_shared = []
        initial_distances = distance.copy()
        groups_used = []
        steps = 0
        
        # Select initial group
        current_group = select_random_group(sim.num_robots, group_sizes, all_groups)
        groups_used.append(current_group)
        
        while steps < max_steps:
            # Switch group every N steps
            if steps > 0 and steps % group_switch_interval == 0:
                current_group = select_random_group(sim.num_robots, group_sizes, all_groups)
                groups_used.append(current_group)
            
            # Prepare state using the policy's prepare_state method
            robot_state, terminal = policy.prepare_state(
                poses, distance, cos, sin, collision, a, goal_positions
            )
            
            # Get action from coupled policy with active group
            action, v_shared, combined_weights = policy.get_action(
                np.array(robot_state), 
                obstacle_states,
                active_group=current_group,
                add_noise=False
            )
            episode_v_shared.append(v_shared)
            
            # Convert action to simulation format
            # action[:, 0] is v_shared in [v_min, v_max] for active robots, 0 for inactive
            # action[:, 1] is omega in [-1, 1] for active robots, 0 for inactive
            # For inactive robots, we need to provide some default action
            a_in = []
            for i, act in enumerate(action):
                if i in current_group:
                    # Active robot: use coupled action
                    a_in.append([act[0], act[1]])
                else:
                    # Inactive robot: stop (or could use decentralized policy)
                    a_in.append([0.0, 0.0])
            
            # Step simulation
            (
                poses, distance, cos, sin, collision, goal, a, reward,
                positions, goal_positions, obstacle_states
            ) = sim.step(a_in, None, combined_weights)
            
            episode_reward += sum(reward)
            episode_goals += sum(goal)
            episode_collisions += sum(collision) / 2  # Avoid double counting
            
            steps += 1
            
            # Check termination conditions
            outside = outside_of_bounds(poses, sim)
            if sum(collision) > 0.5 or outside or all(goal):
                break
        
        # Calculate progress
        final_distances = distance
        total_progress = sum(initial_distances) - sum(final_distances)
        
        metrics.add_episode(
            total_reward=episode_reward,
            goals_reached=episode_goals,
            collisions=episode_collisions,
            steps=steps,
            v_shared_history=episode_v_shared,
            progress=total_progress,
            groups_used=groups_used
        )
        
        if verbose and isinstance(pbar, tqdm):
            summary = metrics.get_summary()
            pbar.set_postfix({
                "goals": f"{summary['avg_goals']:.2f}",
                "cols": f"{summary['avg_collisions']:.2f}",
                "v": f"{summary['avg_v_shared']:.3f}",
                "grp": str(current_group)
            })
    
    return metrics


def main():
    """Main evaluation function."""
    # Load configuration
    config = CONFIG
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create simulation environment
    logger.info("Creating simulation environment with obstacles...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
        reward_phase=5,
        per_robot_goal_reset=True,
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )
    
    logger.info(f"Environment: {sim.num_robots} robots, {sim.num_obstacles} obstacles")
    
    # Create and load policy
    logger.info("Loading group-based coupled action policy...")
    policy = CoupledActionPolicyObstacle(
        state_dim=config["state_dim"],
        obstacle_state_dim=config["obstacle_state_dim"],
        num_robots=config["num_robots"],
        num_obstacles=config["num_obstacles"],
        device=device,
        embedding_dim=config["embedding_dim"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        pooling=config["pooling"],
        load_pretrained_encoder=True,
        pretrained_model_name=config["pretrained_model_name"],
        pretrained_directory=Path(config["pretrained_directory"]),
        freeze_encoder=True,
        freeze_omega=True,
        model_name=config["model_name"],
        save_directory=Path(config["model_directory"]),
    )
    
    # Load trained v_head weights
    try:
        policy.load(filename=config["model_name"], directory=Path(config["model_directory"]))
        logger.info(f"Loaded trained model: {config['model_name']}")
    except FileNotFoundError:
        logger.warning(f"Could not load model {config['model_name']}. Using freshly initialized v_head.")
    
    # Run evaluation
    logger.info(f"Running evaluation for {config['test_scenarios']} episodes...")
    logger.info(f"Group switch interval: every {config['group_switch_interval']} steps")
    logger.info(f"Group sizes: {config['group_sizes']}")
    
    metrics = evaluate_group_policy(
        policy=policy,
        sim=sim,
        num_episodes=config["test_scenarios"],
        max_steps=config["max_steps"],
        group_switch_interval=config["group_switch_interval"],
        group_sizes=config["group_sizes"],
        use_all_combinations=config["use_all_combinations"],
        verbose=True
    )
    
    # Print summary
    summary = metrics.get_summary()
    logger.info("\n" + "=" * 60)
    logger.info("GROUP-BASED COUPLED ACTION POLICY EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Episodes: {summary['num_episodes']}")
    logger.info(f"Average reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
    logger.info(f"Average goals: {summary['avg_goals']:.2f} ± {summary['std_goals']:.2f}")
    logger.info(f"Average collisions: {summary['avg_collisions']:.2f} ± {summary['std_collisions']:.2f}")
    logger.info(f"Collision rate: {summary['collision_rate']:.2%}")
    logger.info(f"Success rate: {summary['success_rate']:.2%}")
    logger.info(f"Average steps: {summary['avg_steps']:.1f}")
    logger.info(f"Average v_shared: {summary['avg_v_shared']:.4f} ± {summary['std_v_shared']:.4f}")
    logger.info(f"Average progress/step: {summary['avg_progress_per_step']:.4f}")
    logger.info("-" * 60)
    logger.info("GROUP STATISTICS:")
    logger.info(f"Total group switches: {summary['total_group_switches']}")
    logger.info(f"Size-2 groups ratio: {summary['group_size_2_ratio']:.2%}")
    logger.info(f"Size-3 groups ratio: {summary['group_size_3_ratio']:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
