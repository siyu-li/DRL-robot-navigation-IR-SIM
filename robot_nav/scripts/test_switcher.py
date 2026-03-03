"""
Test script for evaluating the trained Group Switcher network (v2 - Episode-Level Evaluation).

Key differences from v1:
- NO per-robot reset: all robots reset together when ALL reach their goals.
- Robots continue to receive policy actions after reaching their goal — the
  learned policy naturally keeps them near the goal (no manual zeroing).
- Episode-level success: an episode is "successful" only if ALL robots reach their
  goals AND no collisions occurred during the episode.
- Deterministic seeding: setting the same seed guarantees identical obstacle/robot
  configurations across runs, enabling fair comparison between switcher and random.
- No per-robot trajectory statistics; only episode-level metrics.

Collision statistics (group-level breakdown) remain unchanged from v1.

Usage:
    python -m robot_nav.scripts.test_switcher

Configuration:
    Edit the CONFIG dictionary below to change test settings.
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.switcher.supervised import (
    GroupFeatureBuilder,
    GroupSwitcher,
)
from robot_nav.models.MARL.switcher.rl import (
    RLFeatureBuilder,
    GROUP_SCALAR_DIM,
    STATE_SCALAR_DIM,
    SwitcherActorCritic,
)
from robot_nav.models.MARL.groups.group_generator import (
    generate_all_groups,
    filter_groups_by_size,
)
from robot_nav.models.MARL.groups.action_coupling import actions_for_group
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
# Suppress IRSim warnings - irsim uses loguru, not standard logging
from loguru import logger
logger.disable("irsim")

# =============================================================================
# Configuration Dictionary - Edit these values directly
# =============================================================================
CONFIG = {
    # Selection mode: "switcher", "rl_switcher", or "random"
    "selection_mode": "random",  # Change to "random" for baseline, "rl_switcher" for RL-trained

    # Group selection strategy for switcher mode:
    #   "argmax"  — always select the highest-scoring group (deterministic)
    #   "top_k"   — uniformly random from top k groups (original behavior)
    #   "softmax" — sample from all groups weighted by softmax of scores
    #   "sample"  — RL stochastic policy
    "selection_strategy": "sample",  # Options: "argmax", "top_k", "softmax"

    # Top-k selection: randomly select from top k groups (only used if selection_strategy="top_k")
    # Set to 1 for deterministic (always best), >1 for stochastic selection
    "top_k_selection": 60,
    
    # Softmax temperature: controls randomness in softmax selection (only used if selection_strategy="softmax")
    # Lower temperature (~0.01-0.1) → more deterministic (favors high scores)
    # Higher temperature (1.0+) → more random (more uniform distribution)
    "softmax_temperature": 0.1,

    # Random seed for reproducibility (set to same value for fair comparison)
    "seed": 42,
    
    # Number of trials per episode (same obstacles/start/goal configuration)
    "trials_per_episode": 3,

    # Switcher model configuration (supervised)
    "switcher_checkpoint": "robot_nav/models/MARL/switcher/runs/switcher/epoch_100.pt",

    # RL switcher model configuration (PPO-trained)
    "rl_switcher_checkpoint": "robot_nav/models/MARL/switcher/checkpoint/rl_switcher_14robots/Mar.01/SwitcherPPO-14robots_update2000.pt",

    # Decentralized model configuration (used for all action generation)
    # Pretrained model paths (decentralized TD3Obstacle policy)
    # "decentralized_model_name": "TD3-MARL-obstacle-14robots-gpu_epoch800",
    # "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Feb.10_obstacle_14robot_transfer_gpu",
    "decentralized_model_name": "TD3-MARL-obstacle-14robots",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Mar.02_obstacle_14robot_finetune",

    # Test configuration
    "test_episodes": 50,
    "max_steps_per_episode": 1500,
    "disable_plotting": False,

    # Group selection interval (re-select group every N steps)
    "selection_interval": 10,
    
    # Group generation settings (must match training data)
    "include_size_1": False,         # Include individual robots
    "include_size_2": True,         # Include pairs
    "include_size_3": True,         # Include triplets
    "include_size_4": True,         # Include size-4 groups (rotation-coupled)
    "include_size_7": False,         # Include size-7 groups (rotation-coupled)

    # Policy configuration
    "num_robots": 14,
    "num_obstacles": 7,
    "state_dim": 11,
    "obstacle_state_dim": 4,
    "embedding_dim": 256,
    "v_min": 0.0,
    "v_max": 0.5,
    "pooling": "mean",

    # Switcher feature configuration (must match training)
    # GroupFeatureBuilder config
    #   Features 1-5 are always on (size_feat, coupling_mode, A_in, A_out, A_obs).
    #   extra_group: list of (extra_key, aggregation) for per-group scalars
    #                aggregated over group members. Set to [] to disable.
    #   extra_global: list of extra_key names for global scalars (same for
    #                 every group). Set to [] to disable.
    #   scalar_dim = 5 + len(extra_group) + len(extra_global) + 1 (urgency_flag)
    "max_group_size": 7,
    "rotation_coupling_threshold": 3,
    "extra_group": [
        ("dist_to_goal", "mean"),   # mean_dist_goal_g
        ("dist_to_goal", "min"),    # min_dist_goal_g
        ("clearance",    "min"),    # min_clearance_g
        ("reached",      "mean"),   # frac_reached_g
        ("heading_error","mean"),   # mean_heading_err_g
    ],
    "extra_global": [
        "var_dist_goal_global",     # distance variance (sync signal)
        "frac_reached_global",      # global completion fraction
        "steps_elapsed_frac",       # time pressure
    ],
    
    # Urgency tracking (for stuck robot detection)
    "use_urgency_flag": False,       # Enable urgency flag as additional scalar feature
    "urgency_lookback_window": 20,  # Number of steps to track per robot
    "urgency_stuck_threshold": 0.3, # If robot moved < this distance over lookback, it's stuck

    # World configuration
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    # "world_file": "robot_nav/worlds/multi_robot_world_obstacle.yaml",
    "obstacle_proximity_threshold": 1.5,

    # Device configuration
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# Seeding Utility
# =============================================================================
def set_global_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Helper Functions
# =============================================================================
def generate_candidate_groups(
    num_robots: int,
    include_size_1: bool = True,
    include_size_2: bool = True,
    include_size_3: bool = True,
    include_size_4: bool = False,
    include_size_7: bool = False,
) -> List[List[int]]:
    """
    Generate candidate groups using binary allocation method.

    Uses the group_generator module which produces groups via binary allocation
    rules instead of exhaustive combinatorial enumeration. This is critical for
    14 robots where exhaustive combinations would be too many.

    Args:
        num_robots: Total number of robots.
        include_size_1: Include singletons (individual robots).
        include_size_2: Include pairs.
        include_size_3: Include triplets.
        include_size_4: Include size-4 groups (rotation-coupled).
        include_size_7: Include size-7 groups (rotation-coupled).

    Returns:
        List of robot index groups filtered by the requested sizes.
    """
    if num_robots <= 6:
        m = 3
    elif num_robots <= 14:
        m = 4
    else:
        raise ValueError(
            f"Unsupported number of robots: {num_robots}. "
            f"Binary allocation supports up to 14 robots (m=4)."
        )

    all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)

    # Determine which sizes to include
    allowed_sizes = set()
    if include_size_1:
        allowed_sizes.add(1)
    if include_size_2:
        allowed_sizes.add(2)
    if include_size_3:
        allowed_sizes.add(3)
    if include_size_4:
        allowed_sizes.add(4)
    if include_size_7:
        allowed_sizes.add(7)

    filtered_groups = [g for g in all_groups if len(g) in allowed_sizes]

    return filtered_groups


def outside_of_bounds(poses: List[List[float]], sim: MARL_SIM_OBSTACLE) -> bool:
    """Check if any robot is outside world boundaries."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def robot_outside_bounds(pose: List[float], sim: MARL_SIM_OBSTACLE) -> bool:
    """Check if a specific robot is outside world boundaries."""
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
    """
    Track test evaluation statistics (episode-level).

    An episode is "successful" if and only if ALL robots reach their goals
    with zero collisions during the episode. Otherwise, it is "failed".
    
    Now supports multiple trials per episode configuration.
    """

    # Episode-level metrics (per trial)
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    episode_outcomes: List[str] = field(default_factory=list)
    # "success" = all robots reached goal, no collisions
    # "collision" = at least one collision or out-of-bounds occurred
    # "timeout" = max_steps reached without all robots finishing
    
    # Episode configuration tracking (episode_idx for each trial)
    trial_episode_indices: List[int] = field(default_factory=list)

    # Per-robot path lengths for successful episodes only
    # Each entry is a list of length num_robots with the cumulative path length
    success_episode_path_lengths: List[List[float]] = field(default_factory=list)

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
        steps: int,
        outcome: str,
        episode_idx: int,
        path_lengths: Optional[List[float]] = None,
    ):
        """Record episode-level metrics."""
        self.episode_rewards.append(float(total_reward))
        self.episode_steps.append(int(steps))
        self.episode_outcomes.append(outcome)
        self.trial_episode_indices.append(episode_idx)
        if outcome == "success" and path_lengths is not None:
            self.success_episode_path_lengths.append(list(path_lengths))

    def get_summary(self, trials_per_episode: int = 1) -> Dict:
        """Get summary statistics."""
        num_trials = len(self.episode_rewards)
        num_episodes = len(set(self.trial_episode_indices)) if self.trial_episode_indices else num_trials
        total_executions = sum(self.executed_size_counts.values())

        # Episode statistics (per trial)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0.0

        # Trial-level success / collision / timeout counts
        trial_success_count = sum(1 for o in self.episode_outcomes if o == "success")
        trial_collision_count = sum(1 for o in self.episode_outcomes if o == "collision")
        trial_timeout_count = sum(1 for o in self.episode_outcomes if o == "timeout")

        trial_success_rate = trial_success_count / max(num_trials, 1)
        trial_collision_rate = trial_collision_count / max(num_trials, 1)
        trial_timeout_rate = trial_timeout_count / max(num_trials, 1)
        
        # Episode-level success (at least one trial succeeded)
        episode_success_count = 0
        if self.trial_episode_indices:
            episode_outcomes = {}  # episode_idx -> list of outcomes
            for episode_idx, outcome in zip(self.trial_episode_indices, self.episode_outcomes):
                if episode_idx not in episode_outcomes:
                    episode_outcomes[episode_idx] = []
                episode_outcomes[episode_idx].append(outcome)
            
            for episode_idx, outcomes in episode_outcomes.items():
                if "success" in outcomes:
                    episode_success_count += 1
        
        episode_success_rate = episode_success_count / max(num_episodes, 1)

        # Average rewards by outcome
        success_rewards = [
            r for r, o in zip(self.episode_rewards, self.episode_outcomes) if o == "success"
        ]
        collision_rewards = [
            r for r, o in zip(self.episode_rewards, self.episode_outcomes) if o == "collision"
        ]
        timeout_rewards = [
            r for r, o in zip(self.episode_rewards, self.episode_outcomes) if o == "timeout"
        ]

        avg_success_reward = np.mean(success_rewards) if success_rewards else 0.0
        avg_collision_reward = np.mean(collision_rewards) if collision_rewards else 0.0
        avg_timeout_reward = np.mean(timeout_rewards) if timeout_rewards else 0.0

        # Average steps by outcome
        success_steps = [
            s for s, o in zip(self.episode_steps, self.episode_outcomes) if o == "success"
        ]
        avg_success_steps = np.mean(success_steps) if success_steps else 0.0

        # Path length statistics (only for successful episodes)
        if self.success_episode_path_lengths:
            path_arr = np.array(self.success_episode_path_lengths)  # (num_success, num_robots)
            num_robots = path_arr.shape[1]
            per_robot_avg_path = path_arr.mean(axis=0).tolist()  # avg per robot across success episodes
            per_robot_std_path = path_arr.std(axis=0).tolist()
            total_avg_path = float(path_arr.mean())  # grand mean across all robots & episodes
            total_std_path = float(path_arr.std())
        else:
            num_robots = 0
            per_robot_avg_path = []
            per_robot_std_path = []
            total_avg_path = 0.0
            total_std_path = 0.0

        # Group size distribution
        size_distribution = {}
        for size, count in self.executed_size_counts.items():
            size_distribution[size] = count / max(total_executions, 1)

        # Top 10 executed groups
        top_10_executed = sorted(
            self.executed_group_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Collision rate by group size
        collision_rate_by_size = {}
        for size in [1, 2, 3, 4, 7]:
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
            reverse=True,
        )[:10]

        # Top 10 safest groups
        top_10_safest = sorted(
            collision_rates.items(),
            key=lambda x: x[1],
        )[:10]

        # Collision breakdown by size
        collision_breakdown = {}
        for size in [1, 2, 3, 4, 7]:
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
            "num_trials": num_trials,
            "trials_per_episode": trials_per_episode,
            "total_executions": total_executions,
            "avg_episode_reward": avg_reward,
            "avg_steps": avg_steps,
            # Trial-level metrics
            "trial_success_count": trial_success_count,
            "trial_collision_count": trial_collision_count,
            "trial_timeout_count": trial_timeout_count,
            "trial_success_rate": trial_success_rate,
            "trial_collision_rate": trial_collision_rate,
            "trial_timeout_rate": trial_timeout_rate,
            # Episode-level metrics (at least one trial succeeded)
            "episode_success_count": episode_success_count,
            "episode_success_rate": episode_success_rate,
            "avg_success_reward": avg_success_reward,
            "avg_collision_reward": avg_collision_reward,
            "avg_timeout_reward": avg_timeout_reward,
            "avg_success_steps": avg_success_steps,
            # Path length stats (successful episodes only)
            "per_robot_avg_path_length": per_robot_avg_path,
            "per_robot_std_path_length": per_robot_std_path,
            "total_avg_path_length": total_avg_path,
            "total_std_path_length": total_std_path,
            "num_success_episodes_for_path": len(self.success_episode_path_lengths),
            "size_distribution": size_distribution,
            "top_10_executed_groups": top_10_executed,
            "collision_rate_by_size": collision_rate_by_size,
            "top_10_collision_groups": top_10_collision,
            "top_10_safest_groups": top_10_safest,
            "collision_breakdown_by_size": collision_breakdown,
            "detailed_collision_by_group": detailed_collision_by_group,
        }


# =============================================================================
# Switcher-Based Group Selector (unchanged from v1)
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
        selection_strategy: str = "argmax",
        top_k: int = 1,
        softmax_temperature: float = 1.0,
        use_urgency_flag: bool = True,
    ):
        self.switcher = switcher.to(device)
        self.switcher.eval()
        self.feature_builder = feature_builder
        self.policy = policy
        self.groups = groups
        self.device = device
        self.use_urgency_flag = use_urgency_flag
        
        # Selection strategy
        if selection_strategy not in ["argmax", "top_k", "softmax"]:
            raise ValueError(f"Invalid selection_strategy: {selection_strategy}. "
                           f"Must be 'argmax', 'top_k', or 'softmax'.")
        self.selection_strategy = selection_strategy
        
        # Top-k parameters (only used if strategy="top_k")
        self.top_k = max(1, min(top_k, len(groups)))
        
        # Softmax temperature (only used if strategy="softmax")
        self.softmax_temperature = max(0.01, softmax_temperature)  # Avoid division by zero

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
        embed_dim = H.shape[-1]

        h = H.view(batch_size, n_robots, embed_dim).squeeze(0)
        attn_rr = hard_weights_rr.squeeze(0)
        attn_ro = hard_weights_ro.squeeze(0)

        return h, attn_rr, attn_ro

    def get_extra_features(
        self,
        distance: List[float],
        reached_goal: List[bool],
        poses: List[List[float]],
        goals: List[List[float]],
        sim: MARL_SIM_OBSTACLE,
        current_step: int,
        max_steps: int,
        urgency_flags: Optional[List[bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get extra per-robot and global features.
        
        Per-robot features:
        - dist_to_goal: Distance to goal for each robot
        - clearance: Minimum obstacle clearance for each robot
        - reached: Whether each robot reached its goal (0 or 1)
        - heading_error: Absolute heading error to goal for each robot
        - urgency: Whether each robot is stuck (0 or 1)
        
        Global features:
        - var_dist_goal_global: Variance of distances to goals
        - frac_reached_global: Fraction of robots that reached goals
        - steps_elapsed_frac: Fraction of max steps elapsed
        """
        num_robots = sim.num_robots
        
        # Per-robot features
        dist_to_goal = torch.tensor(distance, dtype=torch.float32, device=self.device)

        clearances = []
        for i in range(num_robots):
            min_clearance = sim.get_min_obstacle_clearance(i)
            clearances.append(min_clearance)
        clearance = torch.tensor(clearances, dtype=torch.float32, device=self.device)
        
        reached = torch.tensor(
            [1.0 if r else 0.0 for r in reached_goal],
            dtype=torch.float32,
            device=self.device
        )
        
        # Heading error: angle difference between current heading and goal direction
        heading_errors = []
        for i in range(num_robots):
            dx = goals[i][0] - poses[i][0]
            dy = goals[i][1] - poses[i][1]
            goal_angle = np.arctan2(dy, dx)
            current_angle = poses[i][2]  # Current heading
            angle_diff = abs(goal_angle - current_angle)
            # Normalize to [0, pi]
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            heading_errors.append(angle_diff)
        heading_error = torch.tensor(heading_errors, dtype=torch.float32, device=self.device)
        
        # Urgency flags (per-robot binary indicator for stuck robots)
        if urgency_flags is None:
            urgency_flags = [False] * num_robots
        urgency = torch.tensor(
            [1.0 if u else 0.0 for u in urgency_flags],
            dtype=torch.float32,
            device=self.device
        )
        
        # Global features (must be tensors, not scalars)
        var_dist_goal_global = torch.var(dist_to_goal)
        frac_reached_global = torch.tensor(
            sum(reached_goal) / max(num_robots, 1),
            dtype=torch.float32,
            device=self.device
        )
        steps_elapsed_frac = torch.tensor(
            current_step / max(max_steps, 1),
            dtype=torch.float32,
            device=self.device
        )

        return {
            # Per-robot features
            "dist_to_goal": dist_to_goal,
            "clearance": clearance,
            "reached": reached,
            "heading_error": heading_error,
            "urgency": urgency,
            # Global features
            "var_dist_goal_global": var_dist_goal_global,
            "frac_reached_global": frac_reached_global,
            "steps_elapsed_frac": steps_elapsed_frac,
        }

    @torch.no_grad()
    def select_group(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        distance: List[float],
        reached_goal: List[bool],
        poses: List[List[float]],
        goals: List[List[float]],
        sim: MARL_SIM_OBSTACLE,
        current_step: int,
        max_steps: int,
        urgency_flags: Optional[List[bool]] = None,
    ) -> List[int]:
        """
        Select a group using the trained switcher.

        Three selection strategies:
        - "argmax": Always select the highest-scoring group (deterministic).
        - "top_k": Uniformly random from top k groups.
        - "softmax": Sample from all groups weighted by softmax of scores.
        """
        h, attn_rr, attn_ro = self.get_embeddings_and_attention(robot_obs, obstacle_obs)
        extra = self.get_extra_features(
            distance, reached_goal, poses, goals, sim, current_step, max_steps,
            urgency_flags=urgency_flags,
        )

        X = self.feature_builder(
            h=h,
            groups=self.groups,
            h_glob=None,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            extra=extra,
        )  # (M, D_base)
        
        # Add urgency flag as additional scalar feature (same as training)
        if self.use_urgency_flag:
            urgency_flags_tensor = self._compute_urgency_flags(self.groups, extra)
            X = torch.cat([X, urgency_flags_tensor.unsqueeze(1).to(self.device)], dim=1)

        X = X.to(self.device)
        logits = self.switcher(X)

        if self.selection_strategy == "argmax":
            # Deterministic: always pick the highest score
            selected_idx = logits.argmax().item()
            
        elif self.selection_strategy == "top_k":
            # Random from top k
            _, top_k_indices = torch.topk(logits, k=self.top_k)
            random_idx = random.randint(0, self.top_k - 1)
            selected_idx = top_k_indices[random_idx].item()
            
        elif self.selection_strategy == "softmax":
            # Softmax sampling: higher scores → higher probability
            # Apply temperature scaling to logits
            scaled_logits = logits / self.softmax_temperature
            probs = torch.softmax(scaled_logits, dim=0)
            
            # Sample from the distribution
            selected_idx = torch.multinomial(probs, num_samples=1).item()
            
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        return self.groups[selected_idx]
    
    def _compute_urgency_flags(
        self,
        groups: List[List[int]],
        extra: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute urgency flags for all groups (same logic as training).
        
        Urgency flag logic:
        - 1.0 if group size == 1 AND that robot is urgent (extra["urgency"][robot_id] == 1.0)
        - 0.0 otherwise (all multi-robot groups OR non-urgent single robots)
        
        Args:
            groups: List of M groups
            extra: Extra features dict containing "urgency" tensor of shape (N,)
            
        Returns:
            urgency_flags: Tensor of shape (M,) with values 0.0 or 1.0
        """
        if "urgency" not in extra:
            # No urgency data available, all flags are 0
            return torch.zeros(len(groups), dtype=torch.float32, device=self.device)
        
        urgency_per_robot = extra["urgency"]  # (N,) with values 0.0 or 1.0
        
        urgency_flags = []
        for group in groups:
            if len(group) == 1:
                # Single-robot group: check if that robot is urgent
                robot_id = group[0]
                is_urgent = urgency_per_robot[robot_id].item()
                urgency_flags.append(is_urgent)
            else:
                # Multi-robot group: urgency flag is always 0
                urgency_flags.append(0.0)
        
        return torch.tensor(urgency_flags, dtype=torch.float32, device=self.device)


# =============================================================================
# RL Switcher-Based Group Selector
# =============================================================================
class RLSwitcherGroupSelector:
    """
    Selects groups using the RL-trained SwitcherActorCritic network (PPO).

    Uses the policy's attention module to get embeddings and attention,
    then builds group features via RLFeatureBuilder and scores them
    using the trained actor (``_actor_logits``).

    Key differences from supervised SwitcherGroupSelector:
    - Uses ``RLFeatureBuilder`` (fixed 13-dim group scalars, separate state features)
      instead of ``GroupFeatureBuilder`` (configurable extra_group/extra_global).
    - Uses ``SwitcherActorCritic._actor_logits(X)`` instead of ``GroupSwitcher(X)``.
    - Extra dict uses ``"var_dist_to_goal"`` key (broadcast N-dim tensor)
      instead of ``"var_dist_goal_global"`` (scalar).
    - No urgency flag (RL feature builder doesn't include it).
    """

    def __init__(
        self,
        actor_critic: SwitcherActorCritic,
        feature_builder: RLFeatureBuilder,
        policy: TD3Obstacle,
        groups: List[List[int]],
        device: torch.device,
        selection_strategy: str = "argmax",
        softmax_temperature: float = 1.0,
    ):
        self.actor_critic = actor_critic.to(device)
        self.actor_critic.eval()
        self.feature_builder = feature_builder
        self.policy = policy
        self.groups = groups
        self.device = device

        if selection_strategy not in ["argmax", "sample", "softmax"]:
            raise ValueError(
                f"Invalid selection_strategy for RL switcher: {selection_strategy}. "
                f"Must be 'argmax', 'sample', or 'softmax'."
            )
        self.selection_strategy = selection_strategy
        self.softmax_temperature = max(0.01, softmax_temperature)

    def _get_embeddings_and_attention(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get robot embeddings and attention weights from the frozen policy."""
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
        embed_dim = H.shape[-1]

        h = H.view(batch_size, n_robots, embed_dim).squeeze(0)
        attn_rr = hard_weights_rr.squeeze(0)
        attn_ro = hard_weights_ro.squeeze(0)

        return h, attn_rr, attn_ro

    def _build_extra(
        self,
        distance: List[float],
        reached_goal: List[bool],
        poses: List[List[float]],
        goals: List[List[float]],
        sim: MARL_SIM_OBSTACLE,
        current_step: int,
        max_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Build extra features dict matching RLFeatureBuilder's expected keys.

        RLFeatureBuilder expects broadcast tensors (shape (N,)) for global scalars,
        and uses key ``"var_dist_to_goal"`` (not ``"var_dist_goal_global"``).
        """
        num_robots = sim.num_robots

        dist_to_goal = torch.tensor(distance, dtype=torch.float32, device=self.device)

        clearances = []
        for i in range(num_robots):
            clearances.append(sim.get_min_obstacle_clearance(i))
        clearance = torch.tensor(clearances, dtype=torch.float32, device=self.device)

        reached = torch.tensor(
            [1.0 if r else 0.0 for r in reached_goal],
            dtype=torch.float32,
            device=self.device,
        )

        heading_errors = []
        for i in range(num_robots):
            dx = goals[i][0] - poses[i][0]
            dy = goals[i][1] - poses[i][1]
            goal_angle = np.arctan2(dy, dx)
            current_angle = poses[i][2]
            angle_diff = abs(goal_angle - current_angle)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            heading_errors.append(angle_diff)
        heading_error = torch.tensor(heading_errors, dtype=torch.float32, device=self.device)

        # Global scalars — RLFeatureBuilder expects broadcast (N,) tensors,
        # accessed via extra["key"][0]
        var_dist_val = torch.var(dist_to_goal).item()
        frac_reached_val = sum(reached_goal) / max(num_robots, 1)
        steps_frac_val = current_step / max(max_steps, 1)

        var_dist_to_goal = torch.full(
            (num_robots,), var_dist_val, dtype=torch.float32, device=self.device
        )
        frac_reached_global = torch.full(
            (num_robots,), frac_reached_val, dtype=torch.float32, device=self.device
        )
        steps_elapsed_frac = torch.full(
            (num_robots,), steps_frac_val, dtype=torch.float32, device=self.device
        )

        return {
            "dist_to_goal": dist_to_goal,
            "clearance": clearance,
            "reached": reached,
            "heading_error": heading_error,
            "var_dist_to_goal": var_dist_to_goal,
            "frac_reached_global": frac_reached_global,
            "steps_elapsed_frac": steps_elapsed_frac,
        }

    @torch.no_grad()
    def select_group(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        distance: List[float],
        reached_goal: List[bool],
        poses: List[List[float]],
        goals: List[List[float]],
        sim: MARL_SIM_OBSTACLE,
        current_step: int,
        max_steps: int,
        urgency_flags: Optional[List[bool]] = None,
    ) -> List[int]:
        """
        Select a group using the RL-trained actor-critic.

        Selection strategies:
        - "argmax": Always select the group with highest logit (greedy).
        - "sample": Sample from the Categorical distribution (as during training).
        - "softmax": Temperature-scaled softmax sampling.
        """
        h, attn_rr, attn_ro = self._get_embeddings_and_attention(robot_obs, obstacle_obs)
        extra = self._build_extra(
            distance, reached_goal, poses, goals, sim, current_step, max_steps
        )

        # Build group features via RLFeatureBuilder.forward() → (M, D)
        group_features = self.feature_builder(
            h=h,
            groups=self.groups,
            h_glob=None,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            extra=extra,
        )

        group_features = group_features.to(self.device)

        # Get actor logits → (M,)
        logits = self.actor_critic._actor_logits(group_features)

        if self.selection_strategy == "argmax":
            selected_idx = logits.argmax().item()

        elif self.selection_strategy == "sample":
            dist = torch.distributions.Categorical(logits=logits)
            selected_idx = dist.sample().item()

        elif self.selection_strategy == "softmax":
            scaled_logits = logits / self.softmax_temperature
            probs = torch.softmax(scaled_logits, dim=0)
            selected_idx = torch.multinomial(probs, num_samples=1).item()

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        return self.groups[selected_idx]


# =============================================================================
# Collision Detection Helper (unchanged from v1)
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

    Returns:
        Dict mapping robot_idx -> collision type ("intra", "extra_robot", "obstacle").
    """
    collision_types = {}
    group_set = set(group)
    num_robots = len(collision)

    for i, is_collided in enumerate(collision):
        if not is_collided:
            continue

        closest_robot_dist = float("inf")
        closest_robot_idx = -1

        for j in range(num_robots):
            if i == j:
                continue
            dist_ij = np.sqrt(
                (poses[i][0] - poses[j][0]) ** 2
                + (poses[i][1] - poses[j][1]) ** 2
            )
            if dist_ij < closest_robot_dist:
                closest_robot_dist = dist_ij
                closest_robot_idx = j

        obstacle_clearance = sim.get_min_obstacle_clearance(i)

        if closest_robot_dist < robot_collision_threshold:
            if i in group_set and closest_robot_idx in group_set:
                collision_types[i] = "intra"
            else:
                collision_types[i] = "extra_robot"
        elif obstacle_clearance < 0.4:
            collision_types[i] = "obstacle"
        else:
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
    rotation_coupling_threshold: int = 3,
) -> List[List[float]]:
    """
    Get action for a specific group using the decentralized policy.

    Delegates to ``robot_nav.models.MARL.groups.action_coupling.actions_for_group``.
    """
    return actions_for_group(
        policy=policy,
        robot_obs=robot_obs,
        obstacle_obs=obstacle_obs,
        group=group,
        num_robots=num_robots,
        rotation_coupling_threshold=rotation_coupling_threshold,
    )


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
    rotation_coupling_threshold: int = 3,
    trials_per_episode: int = 1,
    seed: int = 42,
    verbose: bool = True,
    rl_switcher_selector: Optional[RLSwitcherGroupSelector] = None,
) -> TestStatistics:
    """
    Run test evaluation with either switcher or random group selection.

    Episode semantics (v2):
    - An episode starts with a full environment reset (all robots + obstacles).
    - Robots continue to receive policy actions even after reaching their goal;
      the learned policy naturally keeps them near the goal.
    - The episode ends (success) when ALL robots have reached their goals.
    - The episode ends (collision) if ANY robot collides or goes out of bounds.
    - The episode ends (timeout) if max_steps is reached.
    - After any ending, the environment is fully reset for the next episode.

    Deterministic seeding: before each episode, the RNG is seeded with
    (base_seed + episode_index) so that the same episode index always
    produces the same obstacle/robot configuration regardless of selection mode.
    
    Multiple trials per episode: each episode configuration (obstacles, start, goal)
    is tested trials_per_episode times with different random seeds for action selection.
    """
    stats = TestStatistics()
    num_robots = sim.num_robots
    
    # Urgency tracking configuration
    urgency_lookback_window = CONFIG.get("urgency_lookback_window", 20)
    urgency_stuck_threshold = CONFIG.get("urgency_stuck_threshold", 0.3)

    total_trials = num_episodes * trials_per_episode
    pbar = (
        tqdm(range(num_episodes), desc=f"Testing ({selection_mode})")
        if verbose
        else range(num_episodes)
    )

    for episode in pbar:
        # Run multiple trials with the same environment configuration
        for trial in range(trials_per_episode):
            # ------------------------------------------------------------------
            # Deterministic per-episode seed for environment configuration
            # Use the same seed for all trials of the same episode to get the same
            # obstacle/start/goal configuration
            # ------------------------------------------------------------------
            episode_seed = seed + episode
            set_global_seed(episode_seed)

            # Reset environment with the same seed (same configuration across trials)
            (
                poses, distance, cos, sin, collision, goals, action, reward,
                positions, goal_positions, obstacle_states,
            ) = sim.reset(random_obstacles=True)

            episode_reward = 0.0
            current_group = None
            steps = 0
            episode_had_collision = False

            # Per-robot "reached goal" flags — used only to determine episode success.
            # Robots are NOT made inactive; the policy naturally keeps them near the goal.
            reached_goal = [False] * num_robots

            # Per-robot cumulative path length tracking
            path_lengths = [0.0] * num_robots
            prev_positions = [[poses[i][0], poses[i][1]] for i in range(num_robots)]
            
            # Urgency tracking: maintain a sliding window of recent positions per robot
            robot_position_history = [[] for _ in range(num_robots)]  # List of lists: per-robot position history
            urgency_flags = [False] * num_robots  # Current urgency flags

            while steps < max_steps:
                # ----------------------------------------------------------
                # Update urgency tracking: check if robots have been stuck
                # ----------------------------------------------------------
                for robot_idx in range(num_robots):
                    # Only track urgency for unreached robots
                    if not reached_goal[robot_idx]:
                        # Add current position to history
                        robot_position_history[robot_idx].append([poses[robot_idx][0], poses[robot_idx][1]])
                        
                        # Keep only the most recent lookback_window positions
                        if len(robot_position_history[robot_idx]) > urgency_lookback_window:
                            robot_position_history[robot_idx].pop(0)
                        
                        # Check if robot is stuck (hasn't moved much over the lookback window)
                        if len(robot_position_history[robot_idx]) >= urgency_lookback_window:
                            oldest_pos = robot_position_history[robot_idx][0]
                            current_pos = robot_position_history[robot_idx][-1]
                            displacement = np.linalg.norm(
                                np.array(current_pos) - np.array(oldest_pos)
                            )
                            urgency_flags[robot_idx] = (displacement < urgency_stuck_threshold)
                        else:
                            urgency_flags[robot_idx] = False
                    else:
                        # Already reached robots are not urgent
                        urgency_flags[robot_idx] = False
                
                # ----------------------------------------------------------
                # Check termination: all robots reached goals (success)
                # ----------------------------------------------------------
                if all(reached_goal):
                    break

                # ----------------------------------------------------------
                # Select group at intervals
                # ----------------------------------------------------------
                if steps % selection_interval == 0 or current_group is None:
                    robot_state, _ = policy.prepare_state(
                        poses, distance, cos, sin, collision, action, goal_positions
                    )
                    robot_obs = np.array(robot_state)

                    if selection_mode == "switcher" and switcher_selector is not None:
                        current_group = switcher_selector.select_group(
                            robot_obs=robot_obs,
                            obstacle_obs=obstacle_states,
                            distance=distance,
                            reached_goal=reached_goal,
                            poses=poses,
                            goals=goal_positions,
                            sim=sim,
                            current_step=steps,
                            max_steps=max_steps,
                            urgency_flags=urgency_flags,
                        )
                    elif selection_mode == "rl_switcher" and rl_switcher_selector is not None:
                        current_group = rl_switcher_selector.select_group(
                            robot_obs=robot_obs,
                            obstacle_obs=obstacle_states,
                            distance=distance,
                            reached_goal=reached_goal,
                            poses=poses,
                            goals=goal_positions,
                            sim=sim,
                            current_step=steps,
                            max_steps=max_steps,
                            urgency_flags=urgency_flags,
                        )
                    else:
                        current_group = random.choice(groups)

                    stats.record_group_execution(current_group)

                # ----------------------------------------------------------
                # Get actions (robots at goal get [0,0])
                # ----------------------------------------------------------
                robot_state, _ = policy.prepare_state(
                    poses, distance, cos, sin, collision, action, goal_positions
                )
                robot_obs = np.array(robot_state)

                action_out = get_action_for_group(
                    policy, robot_obs, obstacle_states,
                    current_group, num_robots,
                    rotation_coupling_threshold=rotation_coupling_threshold,
                )

                # ----------------------------------------------------------
                # Step simulation
                # ----------------------------------------------------------
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states,
                ) = sim.step(action_out, None, None)

                steps += 1

                # Accumulate episode reward (all robots)
                episode_reward += sum(reward)

                # Accumulate per-robot path lengths (Euclidean distance moved)
                for i in range(num_robots):
                    dx = poses[i][0] - prev_positions[i][0]
                    dy = poses[i][1] - prev_positions[i][1]
                    path_lengths[i] += np.sqrt(dx * dx + dy * dy)
                    prev_positions[i] = [poses[i][0], poses[i][1]]

                # ----------------------------------------------------------
                # Check for newly reached goals (latch)
                # ----------------------------------------------------------
                for i in range(num_robots):
                    if not reached_goal[i] and goals[i]:
                        reached_goal[i] = True

                # ----------------------------------------------------------
                # Check for collisions / out-of-bounds → episode failure
                # ----------------------------------------------------------
                collided_indices = []
                for i in range(num_robots):
                    if collision[i] or robot_outside_bounds(poses[i], sim):
                        collided_indices.append(i)

                if collided_indices:
                    episode_had_collision = True

                    # Record collision details for current group
                    collision_types = classify_collisions(
                        collision, poses, current_group, sim
                    )
                    stats.record_collision(current_group, collided_indices, collision_types)

                    # Episode ends immediately on collision
                    break

            # ------------------------------------------------------------------
            # Determine trial outcome
            # ------------------------------------------------------------------
            if episode_had_collision:
                outcome = "collision"
            elif all(reached_goal):
                outcome = "success"
            else:
                outcome = "timeout"

            stats.record_episode(
                total_reward=episode_reward,
                steps=steps,
                outcome=outcome,
                episode_idx=episode,
                path_lengths=path_lengths,
            )

        if verbose and isinstance(pbar, tqdm):
            # Calculate running statistics
            running_trial_success = sum(1 for o in stats.episode_outcomes if o == "success")
            running_trial_collision = sum(1 for o in stats.episode_outcomes if o == "collision")
            running_trial_timeout = sum(1 for o in stats.episode_outcomes if o == "timeout")
            n_trials = len(stats.episode_outcomes)
            
            # Calculate episode-level success (at least one trial succeeded)
            episode_outcomes = {}
            for ep_idx, outcome in zip(stats.trial_episode_indices, stats.episode_outcomes):
                if ep_idx not in episode_outcomes:
                    episode_outcomes[ep_idx] = []
                episode_outcomes[ep_idx].append(outcome)
            running_episode_success = sum(1 for outcomes in episode_outcomes.values() if "success" in outcomes)
            n_episodes = len(episode_outcomes)

            pbar.set_postfix({
                "T_S": f"{running_trial_success}/{n_trials}",
                "E_S": f"{running_episode_success}/{n_episodes}",
            })

    return stats


# =============================================================================
# Main
# =============================================================================
def main():
    """Main test function."""
    config = CONFIG

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(config["device"])
    logger.info(f"Using device: {device}")
    logger.info(f"Selection mode: {config['selection_mode']}")
    logger.info(f"Global seed: {config['seed']}")

    # ------------------------------------------------------------------
    # Create simulation environment (per_robot_goal_reset=False so the
    # env does NOT auto-reset individual robot goals on arrival)
    # ------------------------------------------------------------------
    logger.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
        reward_phase=5,
        per_robot_goal_reset=False,  # <-- KEY CHANGE: no per-robot goal reset
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )

    logger.info(f"Environment: {sim.num_robots} robots, {sim.num_obstacles} obstacles")

    # ------------------------------------------------------------------
    # Load decentralized policy
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Generate candidate groups
    # ------------------------------------------------------------------
    groups = generate_candidate_groups(
        num_robots=config["num_robots"],
        include_size_1=config.get("include_size_1", True),
        include_size_2=config.get("include_size_2", True),
        include_size_3=config.get("include_size_3", True),
        include_size_4=config.get("include_size_4", False),
        include_size_7=config.get("include_size_7", False),
    )

    logger.info(f"Candidate groups: {len(groups)} total")
    logger.info(f"  Size-1: {sum(1 for g in groups if len(g) == 1)}")
    logger.info(f"  Size-2: {sum(1 for g in groups if len(g) == 2)}")
    logger.info(f"  Size-3: {sum(1 for g in groups if len(g) == 3)}")
    logger.info(f"  Size-4: {sum(1 for g in groups if len(g) == 4)}")
    logger.info(f"  Size-7: {sum(1 for g in groups if len(g) == 7)}")

    # ------------------------------------------------------------------
    # Setup switcher selector if needed
    # ------------------------------------------------------------------
    switcher_selector = None
    rl_switcher_selector = None
    if config["selection_mode"] == "switcher":
        logger.info("Loading trained switcher...")

        checkpoint_path = Path(config["switcher_checkpoint"])
        if not checkpoint_path.exists():
            logger.error(f"Switcher checkpoint not found: {checkpoint_path}")
            logger.info("Please train the switcher first or use 'random' mode.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint.get("config", {})
        embed_dim = model_config.get("embed_dim", config["embedding_dim"] * 2)

        # Build feature_builder with same config as training
        feature_builder = GroupFeatureBuilder(
            embed_dim=embed_dim,
            max_group_size=config.get("max_group_size", 7),
            rotation_coupling_threshold=config.get("rotation_coupling_threshold", 3),
            extra_group=config.get("extra_group", None),
            extra_global=config.get("extra_global", None),
        )
        
        # scalar_dim is computed by GroupFeatureBuilder
        # If urgency flag is enabled, add 1 to scalar_dim for the model
        base_scalar_dim = feature_builder.scalar_dim
        use_urgency_flag = config.get("use_urgency_flag", True)
        urgency_dim = 1 if use_urgency_flag else 0
        scalar_dim = base_scalar_dim + urgency_dim

        switcher = GroupSwitcher(
            embed_dim=embed_dim,
            scalar_dim=scalar_dim,
            embed_hidden=model_config.get("embed_hidden", 256),
            scalar_hidden=model_config.get("scalar_hidden", 32),
            fusion_hidden=model_config.get("fusion_hidden", 256),
            dropout=model_config.get("dropout", 0.1),
        )

        switcher.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded switcher from {checkpoint_path}")
        logger.info(f"  Embed dim: {embed_dim}, Scalar dim: {scalar_dim}")
        logger.info(f"  Extra group features: {len(config.get('extra_group', []))}")
        logger.info(f"  Extra global features: {len(config.get('extra_global', []))}")
        if use_urgency_flag:
            logger.info(f"  Urgency flag: enabled (adds 1 scalar dimension)")

        switcher_selector = SwitcherGroupSelector(
            switcher=switcher,
            feature_builder=feature_builder,
            policy=policy,
            groups=groups,
            device=device,
            selection_strategy=config.get("selection_strategy", "argmax"),
            top_k=config.get("top_k_selection", 1),
            softmax_temperature=config.get("softmax_temperature", 1.0),
            use_urgency_flag=use_urgency_flag,
        )
        
        # Log the selection strategy
        strategy = config.get("selection_strategy", "argmax")
        if strategy == "argmax":
            logger.info("Switcher selector: deterministic (always best group)")
        elif strategy == "top_k":
            logger.info(f"Switcher selector: random from top {config.get('top_k_selection', 1)} groups")
        elif strategy == "softmax":
            logger.info(f"Switcher selector: softmax sampling (temperature={config.get('softmax_temperature', 1.0)})")

    elif config["selection_mode"] == "rl_switcher":
        logger.info("Loading RL-trained switcher (PPO)...")

        rl_checkpoint_path = Path(config["rl_switcher_checkpoint"])
        if not rl_checkpoint_path.exists():
            logger.error(f"RL switcher checkpoint not found: {rl_checkpoint_path}")
            logger.info("Please train the RL switcher first or use 'random'/'switcher' mode.")
            return

        rl_checkpoint = torch.load(rl_checkpoint_path, map_location=device)

        # Checkpoint saved by SwitcherPPO.save() contains:
        #   "policy_state_dict" — SwitcherActorCritic state dict
        #   "optimizer_state_dict" — Adam optimizer state
        #   "iter_count" — training iteration counter
        if "policy_state_dict" not in rl_checkpoint:
            logger.error(
                f"RL checkpoint missing 'policy_state_dict' key. "
                f"Available keys: {list(rl_checkpoint.keys())}"
            )
            return

        # Determine embed_dim from the checkpoint weights
        # actor_embed_tower.0.weight has shape (embed_hidden, 2*embed_dim)
        embed_tower_weight = rl_checkpoint["policy_state_dict"]["actor_embed_tower.0.weight"]
        embed_dim = embed_tower_weight.shape[1] // 2  # 2*embed_dim → embed_dim

        # Build RLFeatureBuilder
        rl_feature_builder = RLFeatureBuilder(
            embed_dim=embed_dim,
            pooling="mean",
            max_group_size=config.get("max_group_size", 7),
            rotation_coupling_threshold=config.get("rotation_coupling_threshold", 3),
        )

        # Build SwitcherActorCritic with matching architecture
        # Infer hidden dims from checkpoint weight shapes
        sd = rl_checkpoint["policy_state_dict"]
        embed_hidden = sd["actor_embed_tower.0.weight"].shape[0]
        group_scalar_hidden = sd["actor_scalar_tower.0.weight"].shape[0]
        group_scalar_dim = sd["actor_scalar_tower.0.weight"].shape[1]
        fusion_hidden = sd["actor_fusion.0.weight"].shape[0]
        value_embed_hidden = sd["critic_embed_tower.0.weight"].shape[0]
        value_scalar_hidden = sd["critic_scalar_tower.0.weight"].shape[0]
        state_scalar_dim = sd["critic_scalar_tower.0.weight"].shape[1]

        actor_critic = SwitcherActorCritic(
            embed_dim=embed_dim,
            group_scalar_dim=group_scalar_dim,
            state_scalar_dim=state_scalar_dim,
            embed_hidden=embed_hidden,
            group_scalar_hidden=group_scalar_hidden,
            fusion_hidden=fusion_hidden,
            value_embed_hidden=value_embed_hidden,
            value_scalar_hidden=value_scalar_hidden,
        )

        actor_critic.load_state_dict(rl_checkpoint["policy_state_dict"])
        logger.info(f"Loaded RL switcher from {rl_checkpoint_path}")
        logger.info(f"  Embed dim: {embed_dim}")
        logger.info(f"  Group scalar dim: {group_scalar_dim} (expected {GROUP_SCALAR_DIM})")
        logger.info(f"  State scalar dim: {state_scalar_dim} (expected {STATE_SCALAR_DIM})")
        logger.info(f"  Architecture: embed_hidden={embed_hidden}, "
                     f"group_scalar_hidden={group_scalar_hidden}, "
                     f"fusion_hidden={fusion_hidden}")
        if "iter_count" in rl_checkpoint:
            logger.info(f"  Training iterations: {rl_checkpoint['iter_count']}")

        # Map supervised strategies to RL equivalents
        rl_strategy = config.get("selection_strategy", "argmax")
        if rl_strategy == "top_k":
            rl_strategy = "sample"  # top_k doesn't apply to RL; use Categorical sampling
            logger.info("Note: 'top_k' strategy not supported for RL switcher, using 'sample'.")

        rl_switcher_selector = RLSwitcherGroupSelector(
            actor_critic=actor_critic,
            feature_builder=rl_feature_builder,
            policy=policy,
            groups=groups,
            device=device,
            selection_strategy=rl_strategy,
            softmax_temperature=config.get("softmax_temperature", 1.0),
        )

        logger.info(f"RL switcher selector strategy: {rl_strategy}")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    logger.info(f"\nRunning test evaluation for {config['test_episodes']} episodes...")
    logger.info(f"Trials per episode: {config.get('trials_per_episode', 1)}")
    logger.info(f"Total trials: {config['test_episodes'] * config.get('trials_per_episode', 1)}")
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
        rotation_coupling_threshold=config.get("rotation_coupling_threshold", 3),
        trials_per_episode=config.get("trials_per_episode", 1),
        seed=config["seed"],
        verbose=True,
        rl_switcher_selector=rl_switcher_selector,
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    summary = stats.get_summary(trials_per_episode=config.get("trials_per_episode", 1))

    logger.info("\n" + "=" * 70)
    logger.info(f"SWITCHER TEST RESULTS (v2) - Mode: {config['selection_mode'].upper()}")
    logger.info("=" * 70)

    logger.info("\n--- Test Configuration ---")
    logger.info(f"Episodes: {summary['num_episodes']}")
    logger.info(f"Trials per episode: {summary['trials_per_episode']}")
    logger.info(f"Total trials: {summary['num_trials']}")
    logger.info(f"Average episode reward: {summary['avg_episode_reward']:.2f}")
    logger.info(f"Average steps per trial: {summary['avg_steps']:.1f}")

    logger.info("\n" + "=" * 70)
    logger.info("TRIAL-LEVEL RESULTS (Per-Trial Success Rate)")
    logger.info("=" * 70)
    logger.info(f"  Success trials:   {summary['trial_success_count']}/{summary['num_trials']}")
    logger.info(f"  Collision trials: {summary['trial_collision_count']}/{summary['num_trials']}")
    logger.info(f"  Timeout trials:   {summary['trial_timeout_count']}/{summary['num_trials']}")
    logger.info(f"\n  TRIAL SUCCESS RATE:   {summary['trial_success_rate']:.2%}")
    logger.info(f"  TRIAL COLLISION RATE: {summary['trial_collision_rate']:.2%}")
    logger.info(f"  TRIAL TIMEOUT RATE:   {summary['trial_timeout_rate']:.2%}")

    logger.info("\n" + "=" * 70)
    logger.info("EPISODE-LEVEL RESULTS (Success if ANY Trial Succeeds)")
    logger.info("=" * 70)
    logger.info(f"  Success episodes (at least 1 trial succeeded): "
                f"{summary['episode_success_count']}/{summary['num_episodes']}")
    logger.info(f"\n  EPISODE SUCCESS RATE: {summary['episode_success_rate']:.2%}")

    logger.info("\n--- Average Rewards by Outcome ---")
    logger.info(f"  Avg reward (success trials):   {summary['avg_success_reward']:.2f}")
    logger.info(f"  Avg reward (collision trials): {summary['avg_collision_reward']:.2f}")
    logger.info(f"  Avg reward (timeout trials):   {summary['avg_timeout_reward']:.2f}")
    logger.info(f"  Avg steps  (success trials):   {summary['avg_success_steps']:.1f}")

    logger.info("\n--- Path Length Statistics (Success Trials Only) ---")
    if summary["num_success_episodes_for_path"] > 0:
        logger.info(f"  Based on {summary['num_success_episodes_for_path']} successful trials")
        logger.info(f"\n  Per-robot average path length:")
        for i, (avg_pl, std_pl) in enumerate(
            zip(summary["per_robot_avg_path_length"], summary["per_robot_std_path_length"])
        ):
            logger.info(f"    Robot {i:2d}: {avg_pl:.3f} ± {std_pl:.3f}")
        logger.info(
            f"\n  Total average path length (all robots): "
            f"{summary['total_avg_path_length']:.3f} ± {summary['total_std_path_length']:.3f}"
        )
    else:
        logger.info("  No successful trials — path length unavailable.")

    logger.info("\n--- Group Execution Statistics ---")
    logger.info(f"Total group executions: {summary['total_executions']}")

    logger.info("\n--- Group Size Distribution ---")
    for size, pct in sorted(summary["size_distribution"].items()):
        logger.info(f"  Size-{size}: {pct:.1%}")

    logger.info("\n--- Top 10 Executed Groups ---")
    for group, count in summary["top_10_executed_groups"]:
        pct = count / max(summary["total_executions"], 1) * 100
        logger.info(
            f"  {list(group)} (size-{len(group)}): {count} times ({pct:.1f}%)"
        )

    logger.info("\n--- Collision Rate by Group Size ---")
    for size, rate in sorted(summary["collision_rate_by_size"].items()):
        logger.info(f"  Size-{size}: {rate:.2%}")

    logger.info("\n--- Top 10 Groups with Highest Collision Rate ---")
    for group, rate in summary["top_10_collision_groups"]:
        exec_count = stats.group_execution_counts.get(group, 0)
        coll_count = stats.group_collision_counts.get(group, 0)
        logger.info(
            f"  {list(group)} (size-{len(group)}): "
            f"{rate:.2%} ({coll_count}/{exec_count})"
        )

    logger.info("\n--- Top 10 Safest Groups (Lowest Collision Rate) ---")
    for group, rate in summary["top_10_safest_groups"]:
        exec_count = stats.group_execution_counts.get(group, 0)
        coll_count = stats.group_collision_counts.get(group, 0)
        logger.info(
            f"  {list(group)} (size-{len(group)}): "
            f"{rate:.2%} ({coll_count}/{exec_count})"
        )

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

    detailed = summary["detailed_collision_by_group"]
    size_1_groups = {g: d for g, d in detailed.items() if len(g) == 1}
    size_2_groups = {g: d for g, d in detailed.items() if len(g) == 2}
    size_3_groups = {g: d for g, d in detailed.items() if len(g) == 3}
    size_4_groups = {g: d for g, d in detailed.items() if len(g) == 4}
    size_7_groups = {g: d for g, d in detailed.items() if len(g) == 7}

    if size_1_groups:
        logger.info("\n--- Detailed Collision Breakdown for Size-1 Groups ---")
        sorted_size_1 = sorted(
            size_1_groups.items(), key=lambda x: x[1]["total_collisions"], reverse=True
        )
        for group, data in sorted_size_1[:10]:
            if data["total_collisions"] > 0:
                logger.info(
                    f"  {list(group)}: {data['total_collisions']} collisions "
                    f"(extra:{data['extra_robot']}, obs:{data['obstacle']})"
                )

    if size_2_groups:
        logger.info("\n--- Detailed Collision Breakdown for Size-2 Groups ---")
        sorted_size_2 = sorted(
            size_2_groups.items(), key=lambda x: x[1]["total_collisions"], reverse=True
        )
        for group, data in sorted_size_2[:10]:
            if data["total_collisions"] > 0:
                logger.info(
                    f"  {list(group)}: {data['total_collisions']} collisions "
                    f"(intra:{data['intra']}, extra:{data['extra_robot']}, obs:{data['obstacle']})"
                )

    if size_3_groups:
        logger.info("\n--- Detailed Collision Breakdown for Size-3 Groups ---")
        sorted_size_3 = sorted(
            size_3_groups.items(), key=lambda x: x[1]["total_collisions"], reverse=True
        )
        for group, data in sorted_size_3[:10]:
            if data["total_collisions"] > 0:
                logger.info(
                    f"  {list(group)}: {data['total_collisions']} collisions "
                    f"(intra:{data['intra']}, extra:{data['extra_robot']}, obs:{data['obstacle']})"
                )

    if size_4_groups:
        logger.info("\n--- Detailed Collision Breakdown for Size-4 Groups (Rotation-Coupled) ---")
        sorted_size_4 = sorted(
            size_4_groups.items(), key=lambda x: x[1]["total_collisions"], reverse=True
        )
        for group, data in sorted_size_4[:10]:
            if data["total_collisions"] > 0:
                logger.info(
                    f"  {list(group)}: {data['total_collisions']} collisions "
                    f"(intra:{data['intra']}, extra:{data['extra_robot']}, obs:{data['obstacle']})"
                )

    if size_7_groups:
        logger.info("\n--- Detailed Collision Breakdown for Size-7 Groups (Rotation-Coupled) ---")
        sorted_size_7 = sorted(
            size_7_groups.items(), key=lambda x: x[1]["total_collisions"], reverse=True
        )
        for group, data in sorted_size_7[:10]:
            if data["total_collisions"] > 0:
                logger.info(
                    f"  {list(group)}: {data['total_collisions']} collisions "
                    f"(intra:{data['intra']}, extra:{data['extra_robot']}, obs:{data['obstacle']})"
                )

    logger.info("\n" + "=" * 70)

    return stats


if __name__ == "__main__":
    main()
