"""
Lightweight Short-Horizon Oracle for Group Selection Evaluation.

This module implements a short-horizon oracle that evaluates which robot group
(size 1/2/3) is best to activate at a given state by:
1. Simulating forward H steps for each candidate group
2. Accumulating rewards using the existing reward function
3. Selecting the group with the highest cumulative reward

This oracle is for EVALUATION ONLY - to analyze if the environment and policy
favor certain group sizes or compositions.

Key Features:
- Supports individual robots (size-1), pairs (size-2), and triplets (size-3)
- Uses existing low-level policy: 
  - Size-1: decentralized per-robot (v_i, w_i)
  - Size-2/3: coupled action (v_shared, w_i)
- Collects statistics on optimal group selections

Usage:
    python -m robot_nav.coupled_action_oracle_eval
"""

import copy
import logging
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.coupled_action_policy_obstacle import (
    CoupledActionPolicyObstacle,
)
from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE


# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    # Model configuration
    "coupled_model_name": "coupled_action_group_obstacle_best",
    "coupled_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/group_policy",
    "decentralized_model_name": "TD3-MARL-obstacle-6robots_epoch2400",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots_v2",
    
    # Evaluation configuration
    "test_episodes": 20,
    "max_steps_per_episode": 500,
    "disable_plotting": True,
    
    # Oracle configuration
    "oracle_horizon": 10,  # Number of steps to simulate forward
    "oracle_interval": 10,  # Re-evaluate oracle every N steps
    "include_size_1": True,  # Include individual robots as candidates
    "include_size_2": True,  # Include size-2 groups
    "include_size_3": True,  # Include size-3 groups
    
    # Execution mode for real simulation after oracle evaluation
    # "best": Execute with the best group selected by oracle (default)
    # "random": Randomly select a group for execution (unbiased testing)
    # "random_from_top_k": Randomly select from top-k groups by oracle reward
    "execution_mode": "random",
    "top_k_for_random": 3,  # Used when execution_mode is "random_from_top_k"
    
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


# ============================================================================
# Helper Functions
# ============================================================================
def generate_candidate_groups(
    num_robots: int,
    include_size_1: bool = True,
    include_size_2: bool = True,
    include_size_3: bool = True,
) -> List[List[int]]:
    """
    Generate all candidate groups of the specified sizes.
    
    Args:
        num_robots: Total number of robots.
        include_size_1: Include individual robots (decentralized).
        include_size_2: Include pairs.
        include_size_3: Include triplets.
        
    Returns:
        List of robot index groups.
    """
    all_groups = []
    robot_indices = list(range(num_robots))
    
    if include_size_1:
        for i in robot_indices:
            all_groups.append([i])
    
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


# ============================================================================
# Simulation State Snapshot for Rollback
# ============================================================================
@dataclass
class SimulationSnapshot:
    """
    A lightweight snapshot of simulation state for oracle rollouts.
    
    Since we cannot deep-copy the full simulation, we store the essential
    state information needed to restore after hypothetical rollouts.
    """
    robot_states: List[np.ndarray]  # Per-robot [x, y, theta]
    robot_goals: List[np.ndarray]   # Per-robot goal positions
    prev_distances: List[Optional[float]]  # For progress-based reward
    
    @classmethod
    def from_sim(cls, sim: MARL_SIM_OBSTACLE) -> "SimulationSnapshot":
        """Capture current simulation state."""
        robot_states = []
        robot_goals = []
        
        for robot in sim.env.robot_list:
            state = robot.state.copy()
            robot_states.append(state)
            robot_goals.append(robot.goal.copy())
        
        return cls(
            robot_states=robot_states,
            robot_goals=robot_goals,
            prev_distances=sim.prev_distances.copy(),
        )
    
    def restore_to_sim(self, sim: MARL_SIM_OBSTACLE):
        """Restore simulation to this snapshot's state."""
        for i, robot in enumerate(sim.env.robot_list):
            robot.set_state(state=self.robot_states[i], init=True)
            robot.set_goal(self.robot_goals[i], init=True)
        
        sim.prev_distances = self.prev_distances.copy()


# ============================================================================
# Oracle Statistics Tracking
# ============================================================================
@dataclass
class OracleStatistics:
    """
    Track oracle evaluation statistics.
    
    Metrics Explanation:
    - Oracle-level metrics: Recorded each time the oracle evaluates groups (every oracle_interval steps)
    - Episode-level metrics: Accumulated over the ENTIRE episode (all steps until termination)
    
    Reward Calculation:
    - Per-group cumulative reward: Sum of rewards for robots IN THE GROUP over H horizon steps
    - Per-robot reward: Individual robot rewards tracked separately
    - Normalized reward: Per-group reward divided by group size (for fair comparison)
    """
    
    # Counts of how often each group size was selected as best
    size_selection_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Specific group selection counts (tuple of indices -> count)
    group_selection_counts: Dict[Tuple[int, ...], int] = field(default_factory=lambda: defaultdict(int))
    
    # Reward margins (best_reward - second_best_reward)
    reward_margins: List[float] = field(default_factory=list)
    
    # Per-group cumulative rewards across all oracle evaluations (raw sum)
    group_cumulative_rewards: Dict[Tuple[int, ...], List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # Per-group NORMALIZED rewards (reward / group_size) for fair comparison
    group_normalized_rewards: Dict[Tuple[int, ...], List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # Per-robot rewards: robot_index -> list of rewards when that robot was active
    per_robot_rewards: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # Per-robot rewards broken down by group size context
    # (robot_idx, group_size) -> list of that robot's rewards when in a group of that size
    per_robot_rewards_by_group_size: Dict[Tuple[int, int], List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # Episode-level metrics (accumulated over ENTIRE episode, not just one horizon)
    episode_rewards: List[float] = field(default_factory=list)
    episode_goals: List[int] = field(default_factory=list)
    episode_collisions: List[int] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    
    # Track which group was executed (not just oracle-selected) for unbiased analysis
    executed_group_counts: Dict[Tuple[int, ...], int] = field(default_factory=lambda: defaultdict(int))
    executed_size_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Per-group collision tracking across all oracle evaluations
    # group_collision_counts: how many times a collision occurred during oracle rollout for each group
    group_collision_counts: Dict[Tuple[int, ...], int] = field(default_factory=lambda: defaultdict(int))
    # group_evaluation_counts: how many times each group was evaluated by the oracle
    group_evaluation_counts: Dict[Tuple[int, ...], int] = field(default_factory=lambda: defaultdict(int))
    
    def record_oracle_selection(
        self,
        best_group: List[int],
        group_rewards: Dict[Tuple[int, ...], float],
        per_robot_reward_breakdown: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = None,
        group_collisions: Optional[Dict[Tuple[int, ...], bool]] = None,
    ):
        """
        Record an oracle selection decision.
        
        Args:
            best_group: The group selected as best by oracle.
            group_rewards: Dict mapping group tuple to cumulative reward.
            per_robot_reward_breakdown: Optional dict mapping group tuple to 
                {robot_idx: robot's individual reward contribution}.
            group_collisions: Optional dict mapping group tuple to whether collision occurred
                during the oracle rollout for that group.
        """
        group_tuple = tuple(best_group)
        group_size = len(best_group)
        
        self.size_selection_counts[group_size] += 1
        self.group_selection_counts[group_tuple] += 1
        
        # Record all group rewards (both raw and normalized)
        # Convert to float to avoid numpy type coercion issues in statistics.mean()
        for group, reward in group_rewards.items():
            self.group_cumulative_rewards[group].append(float(reward))
            # Normalized by group size for fair comparison
            normalized = float(reward) / len(group) if len(group) > 0 else 0.0
            self.group_normalized_rewards[group].append(normalized)
            # Track evaluation count for collision rate calculation
            self.group_evaluation_counts[group] += 1
        
        # Record per-group collision occurrences
        if group_collisions:
            for group, had_collision in group_collisions.items():
                if had_collision:
                    self.group_collision_counts[group] += 1
        
        # Record per-robot rewards if provided
        if per_robot_reward_breakdown:
            for group, robot_rewards in per_robot_reward_breakdown.items():
                group_size = len(group)
                for robot_idx, robot_reward in robot_rewards.items():
                    self.per_robot_rewards[robot_idx].append(float(robot_reward))
                    self.per_robot_rewards_by_group_size[(robot_idx, group_size)].append(float(robot_reward))
        
        # Calculate margin
        sorted_rewards = sorted(group_rewards.values(), reverse=True)
        if len(sorted_rewards) >= 2:
            margin = float(sorted_rewards[0] - sorted_rewards[1])
            self.reward_margins.append(margin)
    
    def record_executed_group(self, group: List[int]):
        """Record which group was actually executed (may differ from oracle best)."""
        group_tuple = tuple(group)
        self.executed_group_counts[group_tuple] += 1
        self.executed_size_counts[len(group)] += 1
    
    def record_episode(
        self,
        total_reward: float,
        goals_reached: int,
        collisions: int,
        steps: int,
    ):
        """
        Record episode-level metrics.
        
        Note: These are accumulated over the ENTIRE episode (all steps),
        not just a single oracle horizon.
        """
        self.episode_rewards.append(float(total_reward))
        self.episode_goals.append(int(goals_reached))
        self.episode_collisions.append(int(collisions))
        self.episode_steps.append(int(steps))
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total_selections = sum(self.size_selection_counts.values())
        total_executions = sum(self.executed_size_counts.values())
        
        # Size distribution (oracle selections)
        size_dist = {
            size: count / total_selections if total_selections > 0 else 0
            for size, count in self.size_selection_counts.items()
        }
        
        # Size distribution (actual executions)
        executed_size_dist = {
            size: count / total_executions if total_executions > 0 else 0
            for size, count in self.executed_size_counts.items()
        }
        
        # Top 10 most selected groups (by oracle)
        top_groups = sorted(
            self.group_selection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top 10 most executed groups
        top_executed_groups = sorted(
            self.executed_group_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Average RAW reward by group size (larger groups naturally higher)
        avg_raw_reward_by_size = defaultdict(list)
        for group, rewards in self.group_cumulative_rewards.items():
            avg_raw_reward_by_size[len(group)].extend(rewards)
        avg_raw_reward_by_size = {
            size: statistics.mean(rewards) if rewards else 0
            for size, rewards in avg_raw_reward_by_size.items()
        }
        
        # Average NORMALIZED reward by group size (fair comparison)
        avg_normalized_reward_by_size = defaultdict(list)
        for group, rewards in self.group_normalized_rewards.items():
            avg_normalized_reward_by_size[len(group)].extend(rewards)
        avg_normalized_reward_by_size = {
            size: statistics.mean(rewards) if rewards else 0
            for size, rewards in avg_normalized_reward_by_size.items()
        }
        
        # Per-robot average rewards
        per_robot_avg = {
            robot_idx: statistics.mean(rewards) if rewards else 0
            for robot_idx, rewards in self.per_robot_rewards.items()
        }
        
        # Per-robot rewards by group size context
        per_robot_by_size = {}
        for (robot_idx, group_size), rewards in self.per_robot_rewards_by_group_size.items():
            if robot_idx not in per_robot_by_size:
                per_robot_by_size[robot_idx] = {}
            per_robot_by_size[robot_idx][group_size] = statistics.mean(rewards) if rewards else 0
        
        # Per-group collision rates across all oracle evaluations
        # collision_rate = collision_count / evaluation_count for each group
        per_group_collision_rate = {}
        for group, eval_count in self.group_evaluation_counts.items():
            collision_count = self.group_collision_counts.get(group, 0)
            per_group_collision_rate[group] = collision_count / eval_count if eval_count > 0 else 0
        
        # Collision rate aggregated by group size
        collision_rate_by_size = defaultdict(lambda: {"collisions": 0, "evaluations": 0})
        for group, eval_count in self.group_evaluation_counts.items():
            size = len(group)
            collision_count = self.group_collision_counts.get(group, 0)
            collision_rate_by_size[size]["collisions"] += collision_count
            collision_rate_by_size[size]["evaluations"] += eval_count
        collision_rate_by_size = {
            size: counts["collisions"] / counts["evaluations"] if counts["evaluations"] > 0 else 0
            for size, counts in collision_rate_by_size.items()
        }
        
        # Top 10 groups with highest collision rates (for diagnosis)
        top_collision_groups = sorted(
            per_group_collision_rate.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top 10 groups with lowest collision rates (safest groups)
        safest_groups = sorted(
            per_group_collision_rate.items(),
            key=lambda x: x[1],
        )[:10]
        
        return {
            "total_oracle_decisions": total_selections,
            "total_executions": total_executions,
            "oracle_size_distribution": size_dist,
            "executed_size_distribution": executed_size_dist,
            "top_10_oracle_groups": top_groups,
            "top_10_executed_groups": top_executed_groups,
            "avg_raw_reward_by_size": avg_raw_reward_by_size,
            "avg_normalized_reward_by_size": avg_normalized_reward_by_size,
            "per_robot_avg_reward": per_robot_avg,
            "per_robot_reward_by_group_size": per_robot_by_size,
            "per_group_collision_rate": per_group_collision_rate,
            "collision_rate_by_size": collision_rate_by_size,
            "top_10_collision_groups": top_collision_groups,
            "top_10_safest_groups": safest_groups,
            "avg_margin": statistics.mean(self.reward_margins) if self.reward_margins else 0,
            "num_episodes": len(self.episode_rewards),
            "avg_episode_reward": statistics.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_goals": statistics.mean(self.episode_goals) if self.episode_goals else 0,
            "avg_collisions": statistics.mean(self.episode_collisions) if self.episode_collisions else 0,
            "collision_rate": (
                sum(1 for c in self.episode_collisions if c > 0) / len(self.episode_collisions)
                if self.episode_collisions else 0
            ),
            "success_rate": (
                sum(1 for g, c in zip(self.episode_goals, self.episode_collisions) if g > 0 and c == 0)
                / len(self.episode_goals)
                if self.episode_goals else 0
            ),
        }


# ============================================================================
# Short-Horizon Oracle
# ============================================================================
class ShortHorizonOracle:
    """
    Lightweight oracle that evaluates candidate groups via short-horizon simulation.
    
    For each candidate group, simulates forward H steps and accumulates rewards.
    Selects the group with the highest cumulative reward.
    
    Args:
        coupled_policy: CoupledActionPolicyObstacle for size-2/3 groups.
        decentralized_policy: TD3Obstacle for size-1 (individual) groups.
        horizon: Number of steps to simulate forward.
        candidate_groups: List of candidate groups to evaluate.
        device: Torch device.
    """
    
    def __init__(
        self,
        coupled_policy: CoupledActionPolicyObstacle,
        decentralized_policy: TD3Obstacle,
        horizon: int = 5,
        candidate_groups: Optional[List[List[int]]] = None,
        device: torch.device = None,
    ):
        self.coupled_policy = coupled_policy
        self.decentralized_policy = decentralized_policy
        self.horizon = horizon
        self.candidate_groups = candidate_groups or []
        self.device = device or torch.device("cpu")
        
        # Group by size for faster lookup
        self.groups_by_size = defaultdict(list)
        for group in self.candidate_groups:
            self.groups_by_size[len(group)].append(group)
    
    def get_action_for_group(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        group: List[int],
        num_robots: int,
    ) -> List[List[float]]:
        """
        Get action for a specific group using the appropriate policy.
        
        Args:
            robot_obs: Robot observations, shape (num_robots, state_dim).
            obstacle_obs: Obstacle observations, shape (num_obstacles, obs_dim).
            group: List of robot indices in the active group.
            num_robots: Total number of robots.
            
        Returns:
            Actions for all robots, shape (num_robots, 2).
            Inactive robots get [0, 0].
        """
        group_size = len(group)
        
        if group_size == 1:
            # Size-1: Use decentralized policy for the single robot
            robot_idx = group[0]
            action, _ = self.decentralized_policy.get_action(
                robot_obs, obstacle_obs, add_noise=False
            )
            # action is (num_robots, 2) with values in [-1, 1]
            # Scale linear velocity: [-1, 1] -> [0, 0.5] using (v + 1) / 4
            # Angular velocity stays in [-1, 1]
            a_out = []
            for i in range(num_robots):
                if i == robot_idx:
                    scaled_lin_vel = (action[i][0] + 1) / 4  # [-1,1] -> [0,0.5]
                    a_out.append([scaled_lin_vel, action[i][1]])
                else:
                    a_out.append([0.0, 0.0])
            return a_out
        else:
            # Size-2/3: Use coupled action policy
            action, v_shared, _ = self.coupled_policy.get_action(
                robot_obs, obstacle_obs, active_group=group, add_noise=False
            )
            # action already has zeros for inactive robots
            a_out = []
            for i in range(num_robots):
                if i in group:
                    a_out.append([action[i][0], action[i][1]])
                else:
                    a_out.append([0.0, 0.0])
            return a_out
    
    def evaluate_group(
        self,
        sim: MARL_SIM_OBSTACLE,
        policy: CoupledActionPolicyObstacle,
        group: List[int],
        poses: List[List[float]],
        distance: List[float],
        cos: List[float],
        sin: List[float],
        collision: List[bool],
        action: List[List[float]],
        goal_positions: List[List[float]],
        obstacle_states: np.ndarray,
        snapshot: SimulationSnapshot,
    ) -> Tuple[float, bool, Dict[int, float]]:
        """
        Evaluate a group by simulating forward H steps.
        
        Args:
            sim: Simulation environment (will be modified, then restored).
            policy: The coupled action policy.
            group: Robot indices in the group.
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after rollout.
            
        Returns:
            Tuple of (cumulative_reward, had_collision, per_robot_rewards).
            - cumulative_reward: Sum of rewards for all robots in group over H steps
            - had_collision: Whether any robot in group collided
            - per_robot_rewards: Dict mapping robot_idx to that robot's cumulative reward
        """
        cumulative_reward = 0.0
        had_collision = False
        
        # Track per-robot rewards
        per_robot_rewards = {robot_idx: 0.0 for robot_idx in group}
        
        # Current state for rollout
        curr_poses = [p.copy() for p in poses]
        curr_distance = distance.copy()
        curr_cos = cos.copy()
        curr_sin = sin.copy()
        curr_collision = list(collision)
        curr_action = [a.copy() for a in action]
        curr_goal_positions = [g.copy() for g in goal_positions]
        curr_obstacle_states = obstacle_states.copy()
        
        for step in range(self.horizon):
            # Prepare state
            robot_state, _ = policy.prepare_state(
                curr_poses, curr_distance, curr_cos, curr_sin, 
                curr_collision, curr_action, curr_goal_positions
            )
            
            # Get action for this group
            a_in = self.get_action_for_group(
                np.array(robot_state),
                curr_obstacle_states,
                group,
                sim.num_robots,
            )
            
            # Step simulation
            (
                curr_poses, curr_distance, curr_cos, curr_sin,
                curr_collision, curr_goal, curr_action, reward,
                _, curr_goal_positions, curr_obstacle_states
            ) = sim.step(a_in, None, None)
            
            # Accumulate reward for each robot in the group
            for robot_idx in group:
                per_robot_rewards[robot_idx] += reward[robot_idx]
            
            # Accumulate total group reward
            group_reward = sum(reward[i] for i in group)
            cumulative_reward += group_reward
            
            # Check for collision
            if any(curr_collision[i] for i in group):
                had_collision = True
                break
            
            # Check for out of bounds
            if outside_of_bounds(curr_poses, sim):
                had_collision = True
                break
        
        # Restore simulation to original state
        snapshot.restore_to_sim(sim)
        
        return cumulative_reward, had_collision, per_robot_rewards
    
    def select_best_group(
        self,
        sim: MARL_SIM_OBSTACLE,
        policy: CoupledActionPolicyObstacle,
        poses: List[List[float]],
        distance: List[float],
        cos: List[float],
        sin: List[float],
        collision: List[bool],
        action: List[List[float]],
        goal_positions: List[List[float]],
        obstacle_states: np.ndarray,
        execution_mode: str = "best",
        top_k: int = 3,
    ) -> Tuple[List[int], List[int], Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], Dict[int, float]], Dict[Tuple[int, ...], bool]]:
        """
        Evaluate all candidate groups and select one for execution.
        
        Args:
            sim: Simulation environment.
            policy: Coupled action policy.
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            execution_mode: How to select group for execution:
                - "best": Select the group with highest cumulative reward
                - "random": Randomly select any candidate group
                - "random_from_top_k": Randomly select from top-k groups by reward
            top_k: Number of top groups to consider when using "random_from_top_k"
                
        Returns:
            Tuple of (best_group, executed_group, all_group_rewards, per_robot_breakdown, group_collisions).
            - best_group: The group with highest oracle reward
            - executed_group: The group selected for actual execution (based on mode)
            - all_group_rewards: Dict mapping group tuple to cumulative reward
            - per_robot_breakdown: Dict mapping group tuple to {robot_idx: robot_reward}
            - group_collisions: Dict mapping group tuple to whether collision occurred during rollout
        """
        # Take snapshot before rollouts
        snapshot = SimulationSnapshot.from_sim(sim)
        
        group_rewards = {}
        per_robot_breakdown = {}
        group_collisions = {}
        best_group = None
        best_reward = float('-inf')
        
        for group in self.candidate_groups:
            cumulative_reward, had_collision, per_robot_rewards = self.evaluate_group(
                sim, policy, group,
                poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states, snapshot
            )
            
            # Note: Collision penalty is already included in the reward from
            # marl_obstacle_sim.get_reward() (-100.0 * 3 * action[0] for collision)
            # No need to add additional penalty here.
            
            group_rewards[tuple(group)] = cumulative_reward
            per_robot_breakdown[tuple(group)] = per_robot_rewards
            group_collisions[tuple(group)] = had_collision
            
            if cumulative_reward > best_reward:
                best_reward = cumulative_reward
                best_group = group
        
        # Select group for execution based on mode
        if execution_mode == "best":
            executed_group = best_group
        elif execution_mode == "random":
            executed_group = random.choice(self.candidate_groups)
        elif execution_mode == "random_from_top_k":
            # Sort groups by reward and pick randomly from top-k
            sorted_groups = sorted(
                group_rewards.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_k_groups = [list(g) for g, _ in sorted_groups[:min(top_k, len(sorted_groups))]]
            executed_group = random.choice(top_k_groups)
        else:
            raise ValueError(f"Unknown execution_mode: {execution_mode}")
        
        return best_group, executed_group, group_rewards, per_robot_breakdown, group_collisions


# ============================================================================
# Main Evaluation Loop
# ============================================================================
def run_oracle_evaluation(
    sim: MARL_SIM_OBSTACLE,
    coupled_policy: CoupledActionPolicyObstacle,
    decentralized_policy: TD3Obstacle,
    oracle: ShortHorizonOracle,
    num_episodes: int = 20,
    max_steps: int = 500,
    oracle_interval: int = 10,
    execution_mode: str = "best",
    top_k: int = 3,
    verbose: bool = True,
) -> OracleStatistics:
    """
    Run evaluation using the short-horizon oracle for group selection.
    
    Args:
        sim: Simulation environment.
        coupled_policy: Policy for coupled groups (size-2/3).
        decentralized_policy: Policy for individual robots (size-1).
        oracle: ShortHorizonOracle instance.
        num_episodes: Number of test episodes.
        max_steps: Maximum steps per episode.
        oracle_interval: Re-evaluate oracle every N steps.
        execution_mode: How to select group for execution:
            - "best": Execute with oracle's best group
            - "random": Randomly select group (unbiased testing)
            - "random_from_top_k": Randomly from top-k oracle groups
        top_k: Number of top groups for "random_from_top_k" mode.
        verbose: Print progress.
        
    Returns:
        OracleStatistics with evaluation results.
        
    Note on Metrics:
        - Oracle metrics (size_selection_counts, group_rewards): Recorded every oracle_interval steps
        - Executed metrics (executed_size_counts): Which groups actually ran
        - Episode metrics (episode_rewards, etc.): Accumulated over ENTIRE episode
    """
    stats = OracleStatistics()
    
    pbar = tqdm(range(num_episodes), desc="Oracle Eval") if verbose else range(num_episodes)
    
    for episode in pbar:
        # Reset environment
        (
            poses, distance, cos, sin, collision, goal, action, reward,
            positions, goal_positions, obstacle_states
        ) = sim.reset(random_obstacles=True)
        
        episode_reward = 0.0
        episode_goals = 0
        episode_collisions = 0
        current_group = None
        steps = 0
        
        while steps < max_steps:
            # Re-evaluate oracle at intervals or first step
            if steps % oracle_interval == 0:
                best_group, executed_group, group_rewards, per_robot_breakdown, group_collisions = oracle.select_best_group(
                    sim, coupled_policy,
                    poses, distance, cos, sin, collision, action,
                    goal_positions, obstacle_states,
                    execution_mode=execution_mode,
                    top_k=top_k,
                )
                # Record oracle selection (what oracle thinks is best)
                stats.record_oracle_selection(best_group, group_rewards, per_robot_breakdown, group_collisions)
                # Record what we actually execute
                stats.record_executed_group(executed_group)
                current_group = executed_group
            
            # Prepare state
            robot_state, _ = coupled_policy.prepare_state(
                poses, distance, cos, sin, collision, action, goal_positions
            )
            
            # Get action using selected group
            a_in = oracle.get_action_for_group(
                np.array(robot_state),
                obstacle_states,
                current_group,
                sim.num_robots,
            )
            
            # Step simulation (with visualization)
            # Get combined weights for visualization if using coupled policy
            if len(current_group) > 1:
                _, _, combined_weights = coupled_policy.get_action(
                    np.array(robot_state), obstacle_states,
                    active_group=current_group, add_noise=False
                )
            else:
                combined_weights = None
            
            (
                poses, distance, cos, sin, collision, goal, action, reward,
                positions, goal_positions, obstacle_states
            ) = sim.step(a_in, None, combined_weights)
            
            # Accumulate episode-level metrics (over ENTIRE episode)
            episode_reward += sum(reward)
            episode_goals += sum(goal)
            episode_collisions += sum(collision) / 2  # Avoid double counting
            
            steps += 1
            
            # Check termination
            if sum(collision) > 0.5 or outside_of_bounds(poses, sim) or all(goal):
                break
        
        stats.record_episode(
            total_reward=episode_reward,
            goals_reached=episode_goals,
            collisions=int(episode_collisions),
            steps=steps,
        )
        
        if verbose and isinstance(pbar, tqdm):
            summary = stats.get_summary()
            exec_dist = summary.get("executed_size_distribution", {})
            pbar.set_postfix({
                "goals": f"{summary['avg_goals']:.1f}",
                "cols": f"{summary['avg_collisions']:.1f}",
                "s1": f"{exec_dist.get(1, 0):.0%}",
                "s2": f"{exec_dist.get(2, 0):.0%}",
                "s3": f"{exec_dist.get(3, 0):.0%}",
            })
    
    return stats


def main():
    """Main evaluation function."""
    config = CONFIG
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create simulation environment
    logger.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
        reward_phase=5,
        per_robot_goal_reset=True,
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )
    
    logger.info(f"Environment: {sim.num_robots} robots, {sim.num_obstacles} obstacles")
    
    # Load coupled action policy (for size-2/3 groups)
    logger.info("Loading coupled action policy...")
    coupled_policy = CoupledActionPolicyObstacle(
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
        pretrained_model_name=config["decentralized_model_name"],
        pretrained_directory=Path(config["decentralized_model_directory"]),
        freeze_encoder=True,
        freeze_omega=True,
        model_name=config["coupled_model_name"],
        save_directory=Path(config["coupled_model_directory"]),
    )
    
    # Try to load trained v_head
    try:
        coupled_policy.load(
            filename=config["coupled_model_name"],
            directory=Path(config["coupled_model_directory"])
        )
        logger.info(f"Loaded trained coupled model: {config['coupled_model_name']}")
    except FileNotFoundError:
        logger.warning("Could not load coupled model. Using fresh v_head.")
    
    # Load decentralized policy (for size-1 groups)
    logger.info("Loading decentralized policy...")
    decentralized_policy = TD3Obstacle(
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
    
    # Generate candidate groups
    candidate_groups = generate_candidate_groups(
        num_robots=config["num_robots"],
        include_size_1=config["include_size_1"],
        include_size_2=config["include_size_2"],
        include_size_3=config["include_size_3"],
    )
    
    logger.info(f"Candidate groups: {len(candidate_groups)} total")
    logger.info(f"  Size-1: {sum(1 for g in candidate_groups if len(g) == 1)}")
    logger.info(f"  Size-2: {sum(1 for g in candidate_groups if len(g) == 2)}")
    logger.info(f"  Size-3: {sum(1 for g in candidate_groups if len(g) == 3)}")
    
    # Create oracle
    oracle = ShortHorizonOracle(
        coupled_policy=coupled_policy,
        decentralized_policy=decentralized_policy,
        horizon=config["oracle_horizon"],
        candidate_groups=candidate_groups,
        device=device,
    )
    
    # Run evaluation
    logger.info(f"\nRunning oracle evaluation for {config['test_episodes']} episodes...")
    logger.info(f"Oracle horizon: {config['oracle_horizon']} steps")
    logger.info(f"Oracle interval: every {config['oracle_interval']} steps")
    logger.info(f"Execution mode: {config['execution_mode']}")
    if config['execution_mode'] == 'random_from_top_k':
        logger.info(f"Top-k for random selection: {config['top_k_for_random']}")
    
    stats = run_oracle_evaluation(
        sim=sim,
        coupled_policy=coupled_policy,
        decentralized_policy=decentralized_policy,
        oracle=oracle,
        num_episodes=config["test_episodes"],
        max_steps=config["max_steps_per_episode"],
        oracle_interval=config["oracle_interval"],
        execution_mode=config["execution_mode"],
        top_k=config["top_k_for_random"],
        verbose=True,
    )
    
    # Print summary
    summary = stats.get_summary()
    
    logger.info("\n" + "=" * 70)
    logger.info("SHORT-HORIZON ORACLE EVALUATION RESULTS")
    logger.info("=" * 70)
    
    logger.info("\n--- Episode Statistics (Accumulated Over ENTIRE Episode) ---")
    logger.info(f"Episodes: {summary['num_episodes']}")
    logger.info(f"Average episode reward: {summary['avg_episode_reward']:.2f}")
    logger.info(f"Average goals reached: {summary['avg_goals']:.2f}")
    logger.info(f"Average collisions: {summary['avg_collisions']:.2f}")
    logger.info(f"Collision rate: {summary['collision_rate']:.2%}")
    logger.info(f"Success rate: {summary['success_rate']:.2%}")
    
    logger.info("\n--- Oracle Selection Statistics ---")
    logger.info(f"Total oracle decisions: {summary['total_oracle_decisions']}")
    logger.info(f"Total executions: {summary['total_executions']}")
    logger.info(f"Average reward margin (best vs 2nd): {summary['avg_margin']:.3f}")
    
    logger.info("\n--- Oracle's Best Group Size Distribution ---")
    logger.info("(Which group sizes the oracle thinks are best)")
    oracle_dist = summary["oracle_size_distribution"]
    for size in sorted(oracle_dist.keys()):
        logger.info(f"  Size-{size}: {oracle_dist[size]:.1%}")
    
    logger.info("\n--- Actually Executed Group Size Distribution ---")
    logger.info(f"(Execution mode: {config['execution_mode']})")
    exec_dist = summary["executed_size_distribution"]
    for size in sorted(exec_dist.keys()):
        logger.info(f"  Size-{size}: {exec_dist[size]:.1%}")
    
    logger.info("\n--- Average RAW Reward by Group Size (Oracle Rollouts) ---")
    logger.info("(Larger groups naturally have higher raw rewards)")
    raw_by_size = summary["avg_raw_reward_by_size"]
    for size in sorted(raw_by_size.keys()):
        logger.info(f"  Size-{size}: {raw_by_size[size]:.3f}")
    
    logger.info("\n--- Average NORMALIZED Reward by Group Size (Oracle Rollouts) ---")
    logger.info("(Reward / group_size - fair comparison across sizes)")
    norm_by_size = summary["avg_normalized_reward_by_size"]
    for size in sorted(norm_by_size.keys()):
        logger.info(f"  Size-{size}: {norm_by_size[size]:.3f}")
    
    logger.info("\n--- Per-Robot Average Reward (When Active) ---")
    per_robot = summary["per_robot_avg_reward"]
    for robot_idx in sorted(per_robot.keys()):
        logger.info(f"  Robot {robot_idx}: {per_robot[robot_idx]:.3f}")
    
    logger.info("\n--- Per-Robot Reward by Group Size Context ---")
    per_robot_by_size = summary["per_robot_reward_by_group_size"]
    for robot_idx in sorted(per_robot_by_size.keys()):
        size_rewards = per_robot_by_size[robot_idx]
        size_str = ", ".join([f"size-{s}: {r:.3f}" for s, r in sorted(size_rewards.items())])
        logger.info(f"  Robot {robot_idx}: {size_str}")
    
    logger.info("\n--- Top 10 Oracle-Selected Groups ---")
    for group, count in summary["top_10_oracle_groups"]:
        size = len(group)
        pct = count / summary["total_oracle_decisions"] * 100 if summary["total_oracle_decisions"] > 0 else 0
        logger.info(f"  {list(group)} (size-{size}): {count} times ({pct:.1f}%)")
    
    logger.info("\n--- Top 10 Actually Executed Groups ---")
    for group, count in summary["top_10_executed_groups"]:
        size = len(group)
        pct = count / summary["total_executions"] * 100 if summary["total_executions"] > 0 else 0
        logger.info(f"  {list(group)} (size-{size}): {count} times ({pct:.1f}%)")
    
    logger.info("\n--- Collision Rate by Group Size (Oracle Rollouts) ---")
    logger.info("(Percentage of oracle rollouts where collision occurred)")
    collision_by_size = summary["collision_rate_by_size"]
    for size in sorted(collision_by_size.keys()):
        logger.info(f"  Size-{size}: {collision_by_size[size]:.2%}")
    
    logger.info("\n--- Top 10 Groups with Highest Collision Rate (Oracle Rollouts) ---")
    for group, rate in summary["top_10_collision_groups"]:
        eval_count = stats.group_evaluation_counts.get(group, 0)
        collision_count = stats.group_collision_counts.get(group, 0)
        logger.info(f"  {list(group)} (size-{len(group)}): {rate:.2%} ({collision_count}/{eval_count} rollouts)")
    
    logger.info("\n--- Top 10 Safest Groups (Lowest Collision Rate) ---")
    for group, rate in summary["top_10_safest_groups"]:
        eval_count = stats.group_evaluation_counts.get(group, 0)
        collision_count = stats.group_collision_counts.get(group, 0)
        logger.info(f"  {list(group)} (size-{len(group)}): {rate:.2%} ({collision_count}/{eval_count} rollouts)")
    
    logger.info("=" * 70)
    
    return stats


if __name__ == "__main__":
    main()
