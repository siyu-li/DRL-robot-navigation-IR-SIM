"""
Evaluation Script for Coupled Action Policy with Robot Switching.

This script evaluates the coupled action policy where only 3 robots are active at a time.
The active robots are switched according to a predefined schedule.

Usage:
    python -m robot_nav.coupled_action_eval_switching
"""

import logging
import statistics
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.coupled_action_policy import CoupledActionPolicy
from robot_nav.SIM_ENV.marl_sim import MARL_SIM


# ============================================================================
# Configuration Dictionary - Edit these values directly
# ============================================================================
CONFIG = {
    # Model configuration
    "model_name": "coupled_action_supervised_best",
    "model_directory": "robot_nav/models/MARL/marlTD3/checkpoint",
    "pretrained_model_name": "TDR-MARL-train",
    "pretrained_directory": "robot_nav/models/MARL/marlTD3/checkpoint/correct",
    
    # Evaluation configuration
    "test_scenarios": 10,  # Number of test episodes
    "max_steps": 1000,  # Max steps per episode
    "disable_plotting": False,  # Set to True to disable visualization
    
    # Policy configuration
    "num_robots": 5,
    "state_dim": 11,
    "embedding_dim": 256,
    "attention": "igs",  # Options: "igs" or "g2anet"
    "v_min": 0.0,
    "v_max": 0.5,
    
    # World configuration
    "world_file": "robot_nav/worlds/multi_robot_world.yaml",
    
    # Switching configuration
    "switching_schedule": [
        # (step_duration, active_robot_ids)
        (10, [0, 1, 2]),  # First 10 steps: robots 0, 1, 2
        (10, [3, 4, 0]),  # Next 10 steps: robots 3, 4, 0
        (10, [1, 4, 3]),  # Next 10 steps: robots 1, 4, 3
        (10, [2, 3, 4]),  # Next 10 steps: robots 2, 3, 4
    ],
}


def outside_of_bounds(poses: List[List[float]], center=(6, 6), half_size=10.5) -> bool:
    """
    Check if any robot is outside the defined world boundaries.
    
    Args:
        poses: List of [x, y, theta] poses for each robot.
        center: Center of the world bounds.
        half_size: Half-width of the world bounds.
        
    Returns:
        True if any robot is outside bounds.
    """
    for pose in poses:
        norm_x = pose[0] - center[0]
        norm_y = pose[1] - center[1]
        if abs(norm_x) > half_size or abs(norm_y) > half_size:
            return True
    return False


def get_active_robots_at_step(step: int, schedule: List[Tuple[int, List[int]]]) -> List[int]:
    """
    Determine which robots are active at a given step based on the schedule.
    
    Args:
        step: Current step number.
        schedule: List of (duration, robot_ids) tuples.
        
    Returns:
        List of active robot IDs.
    """
    total_steps = 0
    for duration, robot_ids in schedule:
        total_steps += duration
        if step < total_steps:
            return robot_ids
    
    # If step exceeds schedule, cycle through schedule
    schedule_length = sum(duration for duration, _ in schedule)
    step_in_cycle = step % schedule_length
    
    total_steps = 0
    for duration, robot_ids in schedule:
        total_steps += duration
        if step_in_cycle < total_steps:
            return robot_ids
    
    return schedule[0][1]  # Fallback to first schedule


class EvaluationMetrics:
    """Track evaluation metrics across episodes."""
    
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
        self.robot_active_steps = {i: 0 for i in range(5)}  # Track steps each robot was active
        self.robot_goals = {i: 0 for i in range(5)}  # Track goals per robot
    
    def add_episode(
        self,
        total_reward: float,
        goals_reached: int,
        collisions: int,
        steps: int,
        v_shared_history: List[float],
        progress: float,
        robot_active_steps: dict,
        robot_goals: dict
    ):
        """Add metrics for a completed episode."""
        self.episode_rewards.append(total_reward)
        self.episode_goals.append(goals_reached)
        self.episode_collisions.append(collisions)
        self.episode_steps.append(steps)
        self.v_shared_values.extend(v_shared_history)
        self.progress_per_step.append(progress / max(steps, 1))
        
        # Accumulate robot-specific metrics
        for robot_id, count in robot_active_steps.items():
            self.robot_active_steps[robot_id] += count
        for robot_id, count in robot_goals.items():
            self.robot_goals[robot_id] += count
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
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
            "robot_active_steps": self.robot_active_steps,
            "robot_goals": self.robot_goals,
        }


def evaluate_policy_with_switching(
    policy: CoupledActionPolicy,
    sim: MARL_SIM,
    num_episodes: int = 100,
    max_steps: int = 600,
    switching_schedule: List[Tuple[int, List[int]]] = None,
    verbose: bool = True
) -> EvaluationMetrics:
    """
    Evaluate the coupled action policy with robot switching.
    
    Only 3 robots are active at a time, determined by the switching schedule.
    Inactive robots have v=0 and w=0.
    
    Args:
        policy: Trained CoupledActionPolicy.
        sim: MARL simulation environment.
        num_episodes: Number of test episodes.
        max_steps: Maximum steps per episode.
        switching_schedule: List of (duration, robot_ids) tuples defining which robots are active.
        verbose: Print progress.
        
    Returns:
        EvaluationMetrics with recorded results.
    """
    if switching_schedule is None:
        # Default schedule
        switching_schedule = [
            (10, [0, 1, 2]),
            (10, [3, 4, 0]),
            (10, [1, 4, 3]),
        ]
    
    metrics = EvaluationMetrics()
    
    # Initialize connections tensor for visualization
    connections = torch.tensor(
        [[0.0 for _ in range(sim.num_robots - 1)] for _ in range(sim.num_robots)]
    )
    
    pbar = tqdm(range(num_episodes), desc="Evaluating") if verbose else range(num_episodes)
    
    for episode in pbar:
        # Reset environment
        (
            poses, distance, cos, sin, collision, goal, a, reward, 
            positions, goal_positions
        ) = sim.reset()
        
        episode_reward = 0.0
        episode_goals = 0
        episode_collisions = 0
        episode_v_shared = []
        initial_distances = distance.copy()
        episode_robot_active_steps = {i: 0 for i in range(sim.num_robots)}
        episode_robot_goals = {i: 0 for i in range(sim.num_robots)}
        steps = 0
        
        while steps < max_steps:
            # Determine active robots for this step
            active_robot_ids = get_active_robots_at_step(steps, switching_schedule)
            
            # Track active steps per robot
            for robot_id in active_robot_ids:
                episode_robot_active_steps[robot_id] += 1
            
            # Prepare state (all robots for attention network)
            state = policy.prepare_state(
                poses, distance, cos, sin, collision, a, goal_positions
            )
            
            # Get action from coupled policy (for all robots)
            action, v_shared, omega = policy.get_action(np.array(state), add_noise=False)
            episode_v_shared.append(v_shared)
            
            # Create action array with only active robots moving
            a_in = []
            for robot_id in range(sim.num_robots):
                if robot_id in active_robot_ids:
                    # Active robot: use predicted action
                    # action[robot_id, 0] is v_shared, action[robot_id, 1] is omega
                    a_in.append([action[robot_id, 0], action[robot_id, 1]])
                else:
                    # Inactive robot: zero velocity
                    a_in.append([0.0, 0.0])
            
            # Step simulation
            (
                poses, distance, cos, sin, collision, goal, a, reward,
                positions, goal_positions, *rest
            ) = sim.step(a_in, connections)
            
            episode_reward += sum(reward)
            episode_goals += sum(goal)
            episode_collisions += sum(collision) / 2  # Avoid double counting
            
            # Track goals per robot
            for robot_id, g in enumerate(goal):
                if g:
                    episode_robot_goals[robot_id] += 1
            
            steps += 1
            
            # Check termination conditions
            outside = outside_of_bounds(poses)
            if sum(collision) > 0.5 or outside:
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
            robot_active_steps=episode_robot_active_steps,
            robot_goals=episode_robot_goals
        )
        
        if verbose and isinstance(pbar, tqdm):
            summary = metrics.get_summary()
            pbar.set_postfix({
                "goals": f"{summary['avg_goals']:.2f}",
                "cols": f"{summary['avg_collisions']:.2f}",
                "v": f"{summary['avg_v_shared']:.3f}"
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
    logger.info("Creating simulation environment...")
    sim = MARL_SIM(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
        reward_phase=2
    )
    
    # Create and load policy
    logger.info("Loading coupled action policy...")
    policy = CoupledActionPolicy(
        state_dim=config["state_dim"],
        num_robots=config["num_robots"],
        device=device,
        embedding_dim=config["embedding_dim"],
        attention=config["attention"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        pooling="mean",
        load_pretrained_encoder=True,
        pretrained_model_name=config["pretrained_model_name"],
        pretrained_directory=Path(config["pretrained_directory"]),
        freeze_encoder=True,
        freeze_omega=True,
        model_name=config["model_name"],
        save_directory=Path(config["model_directory"])
    )
    
    # Load trained v_head weights
    try:
        policy.load(filename=config["model_name"], directory=Path(config["model_directory"]))
        logger.info(f"Loaded trained model: {config['model_name']}")
    except FileNotFoundError:
        logger.warning(f"Could not load model {config['model_name']}. Using freshly initialized v_head.")
    
    # Log switching schedule
    logger.info("\nRobot Switching Schedule:")
    logger.info("-" * 50)
    cumulative_steps = 0
    for duration, robot_ids in config["switching_schedule"]:
        logger.info(f"Steps {cumulative_steps}-{cumulative_steps + duration - 1}: Robots {robot_ids}")
        cumulative_steps += duration
    logger.info(f"(Pattern repeats every {cumulative_steps} steps)")
    logger.info("-" * 50 + "\n")
    
    # Run evaluation
    logger.info(f"Running evaluation for {config['test_scenarios']} episodes...")
    metrics = evaluate_policy_with_switching(
        policy=policy,
        sim=sim,
        num_episodes=config["test_scenarios"],
        max_steps=config["max_steps"],
        switching_schedule=config["switching_schedule"],
        verbose=True
    )
    
    # Print summary
    summary = metrics.get_summary()
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS (WITH ROBOT SWITCHING)")
    logger.info("=" * 50)
    logger.info(f"Episodes: {summary['num_episodes']}")
    logger.info(f"Average reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
    logger.info(f"Average goals: {summary['avg_goals']:.2f} ± {summary['std_goals']:.2f}")
    logger.info(f"Average collisions: {summary['avg_collisions']:.2f} ± {summary['std_collisions']:.2f}")
    logger.info(f"Collision rate: {summary['collision_rate']:.2%}")
    logger.info(f"Success rate: {summary['success_rate']:.2%}")
    logger.info(f"Average steps: {summary['avg_steps']:.1f}")
    logger.info(f"Average v_shared: {summary['avg_v_shared']:.4f} ± {summary['std_v_shared']:.4f}")
    logger.info(f"Average progress/step: {summary['avg_progress_per_step']:.4f}")
    logger.info("\nPer-Robot Statistics:")
    logger.info("-" * 50)
    for robot_id in range(config["num_robots"]):
        active_steps = summary['robot_active_steps'][robot_id]
        goals = summary['robot_goals'][robot_id]
        logger.info(f"Robot {robot_id}: Active steps={active_steps}, Goals={goals}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()