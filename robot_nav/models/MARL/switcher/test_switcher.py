"""
Test script for evaluating the trained Group Switcher network.

This script evaluates the GroupSwitcher network by comparing its performance
against random group selection. It measures:
- Success rate (percentage of episodes where all robots reach goals)
- Collision rate (percentage of episodes with collisions)
- Average reward (using reward phase 5)
- Group execution statistics
- Detailed collision breakdown (intra-group vs extra-group vs obstacle)

Usage:
    python -m robot_nav.models.MARL.switcher.test_switcher

Configuration:
    Edit the CONFIG dictionary below to change test settings.
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.switcher import GroupFeatureBuilder, GroupSwitcher
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE


# =============================================================================
# Configuration Dictionary - Edit these values directly
# =============================================================================
CONFIG = {
    # Selection mode: "switcher" or "random"
    "selection_mode": "switcher",  # Change to "random" for baseline comparison
    
    # Top-k selection: randomly select from top k groups instead of always best
    # Set to 1 for deterministic (always best), >1 for stochastic selection
    "top_k_selection": 3,
    
    # Switcher model configuration
    "switcher_checkpoint": "robot_nav/models/MARL/switcher/runs/switcher/best.pt",
    
    # Decentralized model configuration (used for all action generation)
    "decentralized_model_name": "TD3-MARL-obstacle-6robots_epoch2400",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots_v2",
    
    # Test configuration
    "test_episodes": 100,
    "max_steps_per_episode": 500,
    "disable_plotting": False,

    # Group selection interval (re-select group every N steps)
    "selection_interval": 10,
    
    # Policy configuration
    "num_robots": 6,
    "num_obstacles": 4,
    "state_dim": 11,
    "obstacle_state_dim": 4,
    "embedding_dim": 256,
    "v_min": 0.0,
    "v_max": 0.5,
    "pooling": "mean",
    
    # Switcher feature configuration (must match training)
    "extra_features": ["dist_to_goal", "clearance"],
    "extra_aggregations": ["mean", "min"],
    
    # World configuration
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle.yaml",
    "obstacle_proximity_threshold": 1.5,
    
    # Device configuration
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# Helper Functions
# =============================================================================
def generate_candidate_groups(
    num_robots: int,
    include_size_2: bool = True,
    include_size_3: bool = True,
) -> List[List[int]]:
    """
    Generate all candidate groups of size 2 and 3.
    
    Args:
        num_robots: Total number of robots.
        include_size_2: Include pairs.
        include_size_3: Include triplets.
        
    Returns:
        List of robot index groups.
    """
    all_groups = []
    robot_indices = list(range(num_robots))
    
    if include_size_2:
        for combo in combinations(robot_indices, 2):
            all_groups.append(list(combo))
    
    if include_size_3:
        for combo in combinations(robot_indices, 3):
            all_groups.append(list(combo))
    
    return all_groups


def outside_of_bounds(poses: List[List[float]], sim: MARL_SIM_OBSTACLE) -> bool:
    """Check if any robot is outside world boundaries."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


# =============================================================================
# Statistics Tracking
# =============================================================================
@dataclass
class TestStatistics:
    """Track test evaluation statistics."""
    
    # Episode-level metrics
    episode_rewards: List[float] = field(default_factory=list)
    episode_goals: List[int] = field(default_factory=list)
    episode_collisions: List[int] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    episode_success: List[bool] = field(default_factory=list)
    
    # Group execution counts
    executed_group_counts: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    executed_size_counts: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    
    # Per-group collision tracking
    group_collision_counts: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_execution_counts: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    
    # Detailed collision breakdown
    group_intra_collisions: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_extra_robot_collisions: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_obstacle_collisions: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    
    def record_group_execution(self, group: List[int]):
        """Record which group was executed."""
        group_tuple = tuple(group)
        self.executed_group_counts[group_tuple] += 1
        self.executed_size_counts[len(group)] += 1
        self.group_execution_counts[group_tuple] += 1
    
    def record_collision(
        self,
        group: List[int],
        collision_indices: List[int],
        collision_types: Dict[int, str],
    ):
        """
        Record collision details.
        
        Args:
            group: The active group when collision occurred.
            collision_indices: Robot indices that collided.
            collision_types: Dict mapping robot_idx -> collision type
                ("intra", "extra_robot", "obstacle")
        """
        group_tuple = tuple(group)
        group_set = set(group)
        
        for robot_idx in collision_indices:
            if robot_idx in group_set:
                self.group_collision_counts[group_tuple] += 1
                
                coll_type = collision_types.get(robot_idx, "unknown")
                if coll_type == "intra":
                    self.group_intra_collisions[group_tuple] += 1
                elif coll_type == "extra_robot":
                    self.group_extra_robot_collisions[group_tuple] += 1
                elif coll_type == "obstacle":
                    self.group_obstacle_collisions[group_tuple] += 1
    
    def record_episode(
        self,
        total_reward: float,
        goals_reached: int,
        collisions: int,
        steps: int,
        success: bool,
    ):
        """Record episode-level metrics."""
        self.episode_rewards.append(float(total_reward))
        self.episode_goals.append(int(goals_reached))
        self.episode_collisions.append(int(collisions))
        self.episode_steps.append(int(steps))
        self.episode_success.append(success)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        num_episodes = len(self.episode_rewards)
        total_executions = sum(self.executed_size_counts.values())
        
        # Episode statistics
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_goals = np.mean(self.episode_goals) if self.episode_goals else 0.0
        avg_collisions = np.mean(self.episode_collisions) if self.episode_collisions else 0.0
        avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0.0
        
        success_rate = np.mean(self.episode_success) if self.episode_success else 0.0
        collision_rate = sum(1 for c in self.episode_collisions if c > 0) / max(num_episodes, 1)
        
        # Group size distribution
        size_distribution = {}
        for size, count in self.executed_size_counts.items():
            size_distribution[size] = count / max(total_executions, 1)
        
        # Top 10 executed groups
        top_10_executed = sorted(
            self.executed_group_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Collision rate by group size
        collision_rate_by_size = {}
        for size in [2, 3]:
            size_groups = [g for g in self.group_execution_counts.keys() if len(g) == size]
            total_exec = sum(self.group_execution_counts.get(g, 0) for g in size_groups)
            total_coll = sum(self.group_collision_counts.get(g, 0) for g in size_groups)
            collision_rate_by_size[size] = total_coll / max(total_exec, 1)
        
        # Top 10 high collision groups
        collision_rates = {}
        for group in self.group_execution_counts.keys():
            exec_count = self.group_execution_counts[group]
            coll_count = self.group_collision_counts.get(group, 0)
            if exec_count > 0:
                collision_rates[group] = coll_count / exec_count
        
        top_10_collision = sorted(
            collision_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top 10 safest groups
        top_10_safest = sorted(
            collision_rates.items(),
            key=lambda x: x[1],
        )[:10]
        
        # Collision breakdown by size
        collision_breakdown = {}
        for size in [2, 3]:
            size_groups = [g for g in self.group_execution_counts.keys() if len(g) == size]
            total_exec = sum(self.group_execution_counts.get(g, 0) for g in size_groups)
            intra = sum(self.group_intra_collisions.get(g, 0) for g in size_groups)
            extra = sum(self.group_extra_robot_collisions.get(g, 0) for g in size_groups)
            obs = sum(self.group_obstacle_collisions.get(g, 0) for g in size_groups)
            
            collision_breakdown[size] = {
                "total_executions": total_exec,
                "intra_group": intra,
                "extra_robot": extra,
                "obstacle": obs,
                "intra_rate": intra / max(total_exec, 1),
                "extra_rate": extra / max(total_exec, 1),
                "obstacle_rate": obs / max(total_exec, 1),
            }
        
        # Detailed collision by group
        detailed_collision_by_group = {}
        for group in self.group_execution_counts.keys():
            exec_count = self.group_execution_counts[group]
            detailed_collision_by_group[group] = {
                "executions": exec_count,
                "total_collisions": self.group_collision_counts.get(group, 0),
                "intra": self.group_intra_collisions.get(group, 0),
                "extra_robot": self.group_extra_robot_collisions.get(group, 0),
                "obstacle": self.group_obstacle_collisions.get(group, 0),
            }
        
        return {
            "num_episodes": num_episodes,
            "total_executions": total_executions,
            "avg_episode_reward": avg_reward,
            "avg_goals": avg_goals,
            "avg_collisions": avg_collisions,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "size_distribution": size_distribution,
            "top_10_executed_groups": top_10_executed,
            "collision_rate_by_size": collision_rate_by_size,
            "top_10_collision_groups": top_10_collision,
            "top_10_safest_groups": top_10_safest,
            "collision_breakdown_by_size": collision_breakdown,
            "detailed_collision_by_group": detailed_collision_by_group,
        }


# =============================================================================
# Switcher-Based Group Selector
# =============================================================================
class SwitcherGroupSelector:
    """
    Selects groups using the trained GroupSwitcher network.
    
    Uses the policy's attention module to get embeddings and attention,
    then builds group features and scores them using the trained switcher.
    """
    
    def __init__(
        self,
        switcher: GroupSwitcher,
        feature_builder: GroupFeatureBuilder,
        policy: TD3Obstacle,
        groups: List[List[int]],
        device: torch.device,
        top_k: int = 1,
    ):
        """
        Args:
            switcher: Trained GroupSwitcher model.
            feature_builder: GroupFeatureBuilder with same config as training.
            policy: TD3Obstacle policy (for getting embeddings).
            groups: List of candidate groups.
            device: Torch device.
            top_k: Randomly select from top k groups (1 = always best, >1 = stochastic).
        """
        self.switcher = switcher.to(device)
        self.switcher.eval()
        self.feature_builder = feature_builder
        self.policy = policy
        self.groups = groups
        self.device = device
        self.top_k = max(1, min(top_k, len(groups)))  # Clamp to valid range
    
    def get_embeddings_and_attention(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get robot embeddings and attention weights from the policy."""
        robot_tensor = torch.tensor(robot_obs, dtype=torch.float32, device=self.device)
        obstacle_tensor = torch.tensor(obstacle_obs, dtype=torch.float32, device=self.device)
        
        if robot_tensor.dim() == 2:
            robot_tensor = robot_tensor.unsqueeze(0)
            obstacle_tensor = obstacle_tensor.unsqueeze(0)
        
        with torch.no_grad():
            (
                H,
                hard_logits_rr, hard_logits_ro,
                dist_rr, dist_ro,
                mean_entropy,
                hard_weights_rr,
                hard_weights_ro,
                combined_weights,
            ) = self.policy.actor.attention(robot_tensor, obstacle_tensor)
        
        batch_size = robot_tensor.shape[0]
        n_robots = robot_tensor.shape[1]
        embed_dim_2 = H.shape[-1]
        
        h = H.view(batch_size, n_robots, embed_dim_2).squeeze(0)
        attn_rr = hard_weights_rr.squeeze(0)
        attn_ro = hard_weights_ro.squeeze(0)
        
        return h, attn_rr, attn_ro
    
    def get_extra_features(
        self,
        distance: List[float],
        sim: MARL_SIM_OBSTACLE,
    ) -> Dict[str, torch.Tensor]:
        """Get extra per-robot features."""
        dist_to_goal = torch.tensor(distance, dtype=torch.float32, device=self.device)
        
        clearances = []
        for i in range(sim.num_robots):
            min_clearance = sim.get_min_obstacle_clearance(i)
            clearances.append(min_clearance)
        clearance = torch.tensor(clearances, dtype=torch.float32, device=self.device)
        
        return {
            "dist_to_goal": dist_to_goal,
            "clearance": clearance,
        }
    
    @torch.no_grad()
    def select_group(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        distance: List[float],
        sim: MARL_SIM_OBSTACLE,
    ) -> List[int]:
        """
        Select a group using the trained switcher.
        
        If top_k > 1, randomly selects from the top k groups.
        If top_k == 1, always selects the best group (deterministic).
        
        Args:
            robot_obs: Robot observations, shape (num_robots, state_dim).
            obstacle_obs: Obstacle observations, shape (num_obstacles, obs_dim).
            distance: Per-robot distances to goal.
            sim: Simulation environment (for extra features).
            
        Returns:
            Selected group (list of robot indices).
        """
        # Get embeddings and attention
        h, attn_rr, attn_ro = self.get_embeddings_and_attention(robot_obs, obstacle_obs)
        
        # Get extra features
        extra = self.get_extra_features(distance, sim)
        
        # Build group features
        X = self.feature_builder(
            h=h,
            groups=self.groups,
            h_glob=None,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            extra=extra,
        )
        
        X = X.to(self.device)
        
        # Get logits from switcher
        logits = self.switcher(X)
        
        # Select from top k groups
        if self.top_k == 1:
            # Deterministic: always select best group
            selected_idx = logits.argmax().item()
        else:
            # Stochastic: randomly select from top k groups
            _, top_k_indices = torch.topk(logits, k=self.top_k)
            # Randomly select one from top k
            random_idx = random.randint(0, self.top_k - 1)
            selected_idx = top_k_indices[random_idx].item()
        
        return self.groups[selected_idx]


# =============================================================================
# Collision Detection Helper
# =============================================================================
def classify_collisions(
    collision: List[bool],
    poses: List[List[float]],
    group: List[int],
    sim: MARL_SIM_OBSTACLE,
    robot_collision_threshold: float = 0.6,
) -> Dict[int, str]:
    """
    Classify collisions by type for each collided robot.
    
    Args:
        collision: Per-robot collision flags.
        poses: Per-robot poses [[x, y, theta], ...].
        group: Active group indices.
        sim: Simulation environment.
        robot_collision_threshold: Distance threshold for robot-robot collision.
        
    Returns:
        Dict mapping robot_idx -> collision type ("intra", "extra_robot", "obstacle").
    """
    collision_types = {}
    group_set = set(group)
    num_robots = len(collision)
    
    for i, is_collided in enumerate(collision):
        if not is_collided:
            continue
        
        # Check for robot-robot collision first
        closest_robot_dist = float('inf')
        closest_robot_idx = -1
        
        for j in range(num_robots):
            if i == j:
                continue
            dist_ij = np.sqrt(
                (poses[i][0] - poses[j][0]) ** 2 +
                (poses[i][1] - poses[j][1]) ** 2
            )
            if dist_ij < closest_robot_dist:
                closest_robot_dist = dist_ij
                closest_robot_idx = j
        
        # Check obstacle distance
        obstacle_clearance = sim.get_min_obstacle_clearance(i)
        
        # Classify collision type
        if closest_robot_dist < robot_collision_threshold:
            # Robot-robot collision
            if i in group_set and closest_robot_idx in group_set:
                collision_types[i] = "intra"
            else:
                collision_types[i] = "extra_robot"
        elif obstacle_clearance < 0.4:  # Obstacle collision threshold
            collision_types[i] = "obstacle"
        else:
            # Default to obstacle if we can't determine
            collision_types[i] = "obstacle"
    
    return collision_types


# =============================================================================
# Action Execution Helper
# =============================================================================
def get_action_for_group(
    policy: TD3Obstacle,
    robot_obs: np.ndarray,
    obstacle_obs: np.ndarray,
    group: List[int],
    num_robots: int,
) -> List[List[float]]:
    """
    Get action for a specific group using the decentralized policy.
    
    This matches the action generation in collect_oracle_data.py:
    - Get actions from decentralized policy (outputs in [-1, 1])
    - Scale linear velocity: [-1, 1] -> [0, 0.5] using (v + 1) / 4
    - For coupled groups (size > 1): use MINIMUM scaled linear velocity
    - Angular velocity: passed through directly (stays in [-1, 1])
    - Inactive robots get [0, 0]
    
    Args:
        policy: TD3Obstacle decentralized policy.
        robot_obs: Robot observations, shape (num_robots, state_dim).
        obstacle_obs: Obstacle observations, shape (num_obstacles, obs_dim).
        group: List of robot indices in the active group.
        num_robots: Total number of robots.
        
    Returns:
        Actions for all robots, shape (num_robots, 2).
        Format: [scaled_lin_vel, angular_vel] where:
            - scaled_lin_vel in [0, 0.5]
            - angular_vel in [-1, 1]
        Inactive robots get [0, 0].
    """
    # Get actions from decentralized policy for all robots
    # action is (num_robots, 2) with values in [-1, 1]
    action, _ = policy.get_action(robot_obs, obstacle_obs, add_noise=False)
    
    # Get scaled linear velocities for robots in the group
    # Scale: [-1, 1] -> [0, 0.5] using (v + 1) / 4
    scaled_lin_vels = []
    for idx in group:
        scaled_lin_vel = (action[idx][0] + 1) / 4  # [-1,1] -> [0,0.5]
        scaled_lin_vels.append(scaled_lin_vel)
    
    # Use minimum linear velocity as the coupled velocity
    # This ensures safety - coupled robots move at the slowest robot's speed
    v_coupled = min(scaled_lin_vels)
    
    # Build output actions
    a_out = []
    for i in range(num_robots):
        if i in group:
            # Use coupled linear velocity, individual angular velocity
            a_out.append([v_coupled, action[i][1]])
        else:
            a_out.append([0.0, 0.0])
    
    return a_out


# =============================================================================
# Main Evaluation Loop
# =============================================================================
def run_test_evaluation(
    sim: MARL_SIM_OBSTACLE,
    policy: TD3Obstacle,
    groups: List[List[int]],
    selection_mode: str,
    switcher_selector: Optional[SwitcherGroupSelector],
    num_episodes: int = 100,
    max_steps: int = 500,
    selection_interval: int = 10,
    verbose: bool = True,
) -> TestStatistics:
    """
    Run test evaluation with either switcher or random group selection.
    
    Args:
        sim: Simulation environment.
        policy: TD3Obstacle policy.
        groups: List of candidate groups.
        selection_mode: "switcher" or "random".
        switcher_selector: SwitcherGroupSelector instance (required if mode is "switcher").
        num_episodes: Number of test episodes.
        max_steps: Maximum steps per episode.
        selection_interval: Re-select group every N steps.
        verbose: Print progress.
        
    Returns:
        TestStatistics with evaluation results.
    """
    stats = TestStatistics()
    num_robots = sim.num_robots
    
    pbar = tqdm(range(num_episodes), desc=f"Testing ({selection_mode})") if verbose else range(num_episodes)
    
    for episode in pbar:
        # Reset environment with random obstacles
        (
            poses, distance, cos, sin, collision, goals, action, reward,
            positions, goal_positions, obstacle_states
        ) = sim.reset(random_obstacles=True)
        
        episode_reward = 0.0
        episode_goals = 0
        episode_collisions = 0
        current_group = None
        steps = 0
        goals_reached = [False] * num_robots
        
        while steps < max_steps:
            # Select group at intervals
            if steps % selection_interval == 0 or current_group is None:
                # Prepare robot observations
                robot_state, _ = policy.prepare_state(
                    poses, distance, cos, sin, collision, action, goal_positions
                )
                robot_obs = np.array(robot_state)
                
                if selection_mode == "switcher" and switcher_selector is not None:
                    current_group = switcher_selector.select_group(
                        robot_obs, obstacle_states, distance, sim
                    )
                else:
                    # Random selection
                    current_group = random.choice(groups)
                
                stats.record_group_execution(current_group)
            
            # Get action for current group
            robot_state, _ = policy.prepare_state(
                poses, distance, cos, sin, collision, action, goal_positions
            )
            robot_obs = np.array(robot_state)
            
            action_out = get_action_for_group(
                policy, robot_obs, obstacle_states, current_group, num_robots
            )
            
            # Step simulation
            (
                poses, distance, cos, sin, collision, goals,
                action, reward, positions, goal_positions, obstacle_states
            ) = sim.step(action_out, None, None)
            
            steps += 1
            
            # Accumulate reward (sum of per-robot rewards for robots in group)
            for i in current_group:
                episode_reward += reward[i]
            
            # Track goals reached
            for i, g in enumerate(goals):
                if g and not goals_reached[i]:
                    goals_reached[i] = True
                    episode_goals += 1
            
            # Check for collision
            if any(collision):
                episode_collisions += 1
                
                # Classify collision types
                collision_types = classify_collisions(
                    collision, poses, current_group, sim
                )
                collision_indices = [i for i, c in enumerate(collision) if c]
                stats.record_collision(current_group, collision_indices, collision_types)
                
                # Reset on collision
                (
                    poses, distance, cos, sin, collision, goals, action, reward,
                    positions, goal_positions, obstacle_states
                ) = sim.reset(random_obstacles=True)
                goals_reached = [False] * num_robots
                current_group = None
                continue
            
            # Check for out of bounds
            if outside_of_bounds(poses, sim):
                episode_collisions += 1
                
                # Reset on out of bounds
                (
                    poses, distance, cos, sin, collision, goals, action, reward,
                    positions, goal_positions, obstacle_states
                ) = sim.reset(random_obstacles=True)
                goals_reached = [False] * num_robots
                current_group = None
                continue
            
            # Check if all goals reached
            if all(goals_reached):
                break
        
        # Record episode statistics
        success = all(goals_reached)
        stats.record_episode(
            total_reward=episode_reward,
            goals_reached=episode_goals,
            collisions=episode_collisions,
            steps=steps,
            success=success,
        )
        
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({
                "reward": f"{episode_reward:.1f}",
                "goals": episode_goals,
                "collisions": episode_collisions,
                "success": success,
            })
    
    return stats


# =============================================================================
# Main
# =============================================================================
def main():
    """Main test function."""
    config = CONFIG
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device(config["device"])
    logger.info(f"Using device: {device}")
    logger.info(f"Selection mode: {config['selection_mode']}")
    
    # Create simulation environment
    logger.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
        reward_phase=5,  # Use reward phase 5 for evaluation
        per_robot_goal_reset=True,
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )
    
    logger.info(f"Environment: {sim.num_robots} robots, {sim.num_obstacles} obstacles")
    
    # Load decentralized policy
    logger.info("Loading decentralized policy...")
    policy = TD3Obstacle(
        state_dim=config["state_dim"],
        action_dim=2,
        max_action=1.0,
        device=device,
        num_robots=config["num_robots"],
        num_obstacles=config["num_obstacles"],
        obstacle_state_dim=config["obstacle_state_dim"],
        load_model=True,
        model_name=config["decentralized_model_name"],
        load_model_name=config["decentralized_model_name"],
        load_directory=Path(config["decentralized_model_directory"]),
        save_directory=Path(config["decentralized_model_directory"]),
    )
    
    # Generate candidate groups (size 2 and 3 only)
    groups = generate_candidate_groups(
        num_robots=config["num_robots"],
        include_size_2=True,
        include_size_3=True,
    )
    
    logger.info(f"Candidate groups: {len(groups)} total")
    logger.info(f"  Size-2: {sum(1 for g in groups if len(g) == 2)}")
    logger.info(f"  Size-3: {sum(1 for g in groups if len(g) == 3)}")
    
    # Setup switcher selector if needed
    switcher_selector = None
    if config["selection_mode"] == "switcher":
        logger.info("Loading trained switcher...")
        
        # Load checkpoint
        checkpoint_path = Path(config["switcher_checkpoint"])
        if not checkpoint_path.exists():
            logger.error(f"Switcher checkpoint not found: {checkpoint_path}")
            logger.info("Please train the switcher first or use 'random' mode.")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model config from checkpoint
        model_config = checkpoint.get("config", {})
        embed_dim = model_config.get("embed_dim", config["embedding_dim"] * 2)  # Default to 512
        
        # Infer scalar_dim from extra features
        extra_features = config["extra_features"]
        scalar_dim = 1 + 3 + len(extra_features)  # size + attn_stats + extras
        
        # Create feature builder
        feature_builder = GroupFeatureBuilder(
            embed_dim=embed_dim,
            extra_feature_names=extra_features,
            extra_aggregations=config["extra_aggregations"],
        )
        
        # Create switcher model
        switcher = GroupSwitcher(
            embed_dim=embed_dim,
            scalar_dim=scalar_dim,
            embed_hidden=model_config.get("embed_hidden", 256),
            scalar_hidden=model_config.get("scalar_hidden", 32),
            fusion_hidden=model_config.get("fusion_hidden", 256),
            dropout=model_config.get("dropout", 0.1),
        )
        
        # Load weights
        switcher.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded switcher from {checkpoint_path}")
        
        # Create selector
        switcher_selector = SwitcherGroupSelector(
            switcher=switcher,
            feature_builder=feature_builder,
            policy=policy,
            groups=groups,
            device=device,
            top_k=config.get("top_k_selection", 1),
        )
        logger.info(f"Switcher selector will randomly choose from top {config.get('top_k_selection', 1)} groups")
    
    # Run evaluation
    logger.info(f"\nRunning test evaluation for {config['test_episodes']} episodes...")
    logger.info(f"Selection interval: every {config['selection_interval']} steps")
    
    stats = run_test_evaluation(
        sim=sim,
        policy=policy,
        groups=groups,
        selection_mode=config["selection_mode"],
        switcher_selector=switcher_selector,
        num_episodes=config["test_episodes"],
        max_steps=config["max_steps_per_episode"],
        selection_interval=config["selection_interval"],
        verbose=True,
    )
    
    # Print summary
    summary = stats.get_summary()
    
    logger.info("\n" + "=" * 70)
    logger.info(f"SWITCHER TEST RESULTS - Mode: {config['selection_mode'].upper()}")
    logger.info("=" * 70)
    
    logger.info("\n--- Episode Statistics ---")
    logger.info(f"Episodes: {summary['num_episodes']}")
    logger.info(f"Average episode reward: {summary['avg_episode_reward']:.2f}")
    logger.info(f"Average goals reached: {summary['avg_goals']:.2f}")
    logger.info(f"Average collisions: {summary['avg_collisions']:.2f}")
    logger.info(f"Average steps: {summary['avg_steps']:.1f}")
    logger.info(f"Success rate: {summary['success_rate']:.2%}")
    logger.info(f"Collision rate: {summary['collision_rate']:.2%}")
    
    logger.info("\n--- Group Execution Statistics ---")
    logger.info(f"Total group executions: {summary['total_executions']}")
    
    logger.info("\n--- Group Size Distribution ---")
    for size, pct in sorted(summary["size_distribution"].items()):
        logger.info(f"  Size-{size}: {pct:.1%}")
    
    logger.info("\n--- Top 10 Executed Groups ---")
    for group, count in summary["top_10_executed_groups"]:
        pct = count / max(summary["total_executions"], 1) * 100
        logger.info(f"  {list(group)} (size-{len(group)}): {count} times ({pct:.1f}%)")
    
    logger.info("\n--- Collision Rate by Group Size ---")
    for size, rate in sorted(summary["collision_rate_by_size"].items()):
        logger.info(f"  Size-{size}: {rate:.2%}")
    
    logger.info("\n--- Top 10 Groups with Highest Collision Rate ---")
    for group, rate in summary["top_10_collision_groups"]:
        exec_count = stats.group_execution_counts.get(group, 0)
        coll_count = stats.group_collision_counts.get(group, 0)
        logger.info(f"  {list(group)} (size-{len(group)}): {rate:.2%} ({coll_count}/{exec_count})")
    
    logger.info("\n--- Top 10 Safest Groups (Lowest Collision Rate) ---")
    for group, rate in summary["top_10_safest_groups"]:
        exec_count = stats.group_execution_counts.get(group, 0)
        coll_count = stats.group_collision_counts.get(group, 0)
        logger.info(f"  {list(group)} (size-{len(group)}): {rate:.2%} ({coll_count}/{exec_count})")
    
    logger.info("\n" + "=" * 70)
    logger.info("DETAILED COLLISION BREAKDOWN (Intra-Group vs Extra-Group vs Obstacle)")
    logger.info("=" * 70)
    
    logger.info("\n--- Collision Type Breakdown by Group Size ---")
    for size, breakdown in sorted(summary["collision_breakdown_by_size"].items()):
        total_exec = breakdown["total_executions"]
        logger.info(f"\n  Size-{size} ({total_exec} executions):")
        logger.info(f"    Intra-group:  {breakdown['intra_group']:4d} ({breakdown['intra_rate']:.2%})")
        logger.info(f"    Extra-robot:  {breakdown['extra_robot']:4d} ({breakdown['extra_rate']:.2%})")
        logger.info(f"    Obstacle:     {breakdown['obstacle']:4d} ({breakdown['obstacle_rate']:.2%})")
    
    # Detailed breakdown for size-2 groups
    detailed = summary["detailed_collision_by_group"]
    size_2_groups = {g: d for g, d in detailed.items() if len(g) == 2}
    size_3_groups = {g: d for g, d in detailed.items() if len(g) == 3}
    
    if size_2_groups:
        logger.info("\n--- Detailed Collision Breakdown for Size-2 Groups ---")
        sorted_size_2 = sorted(size_2_groups.items(), key=lambda x: x[1]["total_collisions"], reverse=True)
        for group, data in sorted_size_2[:10]:
            if data["total_collisions"] > 0:
                logger.info(f"  {list(group)}: {data['total_collisions']} collisions "
                           f"(intra:{data['intra']}, extra:{data['extra_robot']}, obs:{data['obstacle']})")
    
    if size_3_groups:
        logger.info("\n--- Detailed Collision Breakdown for Size-3 Groups ---")
        sorted_size_3 = sorted(size_3_groups.items(), key=lambda x: x[1]["total_collisions"], reverse=True)
        for group, data in sorted_size_3[:10]:
            if data["total_collisions"] > 0:
                logger.info(f"  {list(group)}: {data['total_collisions']} collisions "
                           f"(intra:{data['intra']}, extra:{data['extra_robot']}, obs:{data['obstacle']})")
    
    logger.info("\n" + "=" * 70)
    
    return stats


if __name__ == "__main__":
    main()
