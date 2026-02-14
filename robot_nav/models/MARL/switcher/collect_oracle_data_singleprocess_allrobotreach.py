"""
Oracle Data Collection Script for Group Switcher Training.

This script collects training data for the GroupSwitcher by running
oracle evaluations (simulation rollouts) for different group selections.

The oracle evaluates each candidate group by:
1. Simulating forward N steps using the appropriate policy
2. Accumulating rewards over the horizon
3. Averaging over multiple rollouts
4. Early termination if collision occurs

The collected data format is compatible with train_switcher.py.

Usage:
    python -m robot_nav.models.MARL.switcher.collect_oracle_data

Data Collection Methods:
------------------------
1. SIMULATION ROLLOUTS: For each candidate group, run multiple rollouts and
   measure success rate, collision rate, time to goal, etc.

2. EXPERT DEMONSTRATIONS: Have an expert label which group is best, or rank
   the groups for each scenario.

3. REWARD-BASED: Use cumulative reward from RL environment as the score.
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
# Suppress IRSim warnings - irsim uses loguru, not standard logging
from loguru import logger
logger.disable("irsim")

from robot_nav.models.MARL.switcher.group_generator import (
    generate_all_groups,
    filter_groups_by_size,
)

# =============================================================================
# Configuration Dictionary - Edit these values directly
# =============================================================================
CONFIG = {
    # Output configuration
    "output_path": "robot_nav/models/MARL/switcher/data/oracle_data_14robots.pt",
    
    # Data collection settings
    "n_samples": 7000,              # Number of samples to collect
    "n_robots": 14,                  # Number of robots
    "n_obstacles": 7,               # Number of obstacles
    "embed_dim": 256,               # Embedding dimension from GAT backbone
    "seed": 42,                     # Random seed for reproducibility
    
    # Oracle evaluation settings
    "oracle_horizon": 10,           # Number of steps to simulate forward for each group
    "n_rollouts_per_group": 1,      # Number of rollouts to average for each group score
    
    # Group generation settings
    "include_size_1": True,         # Include individual robots as candidates
    "include_size_2": True,         # Include pairs
    "include_size_3": True,         # Include triplets
    
    # Model configuration
    "state_dim": 11,
    "obstacle_state_dim": 4,
    
    # Pretrained model paths (decentralized TD3Obstacle policy)
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-gpu_epoch800",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Feb.10_obstacle_14robot_transfer_gpu",

    # Simulation settings
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "disable_plotting": False,
    "obstacle_proximity_threshold": 1.5,
    "max_steps_per_episode": 800,   # Reset episode after this many steps
    
    # Goal reach threshold (matches world yaml goal_threshold)
    "goal_reach_threshold": 0.3,
    
    # Scoring weights
    "k_reach": 50.0,               # New-reach bonus weight
    "k_progress": 3.0,             # Progress reward weight
    "k_sync": 8.0,                 # Synchronization reward weight
    
    # Debug mode: enables plotting, prints per-group score breakdowns,
    # and pauses after each sample for manual inspection.
    "debug_mode": True,
    
    # Phase 2 group selection strategy:
    #   "random"  — uniformly random group (original behavior)
    #   "top_k"   — randomly sample from the top-k scoring groups (from Phase 1 oracle scores)
    "phase2_selection": "top_k",
    "phase2_top_k": 10,             # Number of top groups to sample from when using "top_k"
}


# =============================================================================
# Helper Functions
# =============================================================================
def generate_candidate_groups(
    num_robots: int,
    include_size_1: bool = True,
    include_size_2: bool = True,
    include_size_3: bool = True,
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

    Returns:
        List of robot index groups with size <= 3.
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

    min_size = 1 if include_size_1 else 2
    max_size = 3 if include_size_3 else (2 if include_size_2 else 1)

    filtered_groups = filter_groups_by_size(all_groups, min_size=min_size, max_size=max_size)

    if not include_size_1:
        filtered_groups = [g for g in filtered_groups if len(g) > 1]
    if not include_size_2:
        filtered_groups = [g for g in filtered_groups if len(g) != 2]
    if not include_size_3:
        filtered_groups = [g for g in filtered_groups if len(g) < 3]

    return filtered_groups


def outside_of_bounds(poses: List[List[float]], sim) -> bool:
    """Check if any robot is outside world boundaries."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


# =============================================================================
# Simulation State Snapshot for Rollback
# =============================================================================
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
    def from_sim(cls, sim) -> "SimulationSnapshot":
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
    
    def restore_to_sim(self, sim):
        """Restore simulation to this snapshot's state."""
        for i, robot in enumerate(sim.env.robot_list):
            robot.set_state(state=self.robot_states[i], init=True)
            robot.set_goal(self.robot_goals[i], init=True)
        sim.prev_distances = self.prev_distances.copy()


# =============================================================================
# Oracle Data Collector
# =============================================================================
class OracleDataCollector:
    """
    Collects oracle data by running simulation rollouts.
    
    For each candidate group, simulates forward H steps and accumulates rewards.
    Uses only the decentralized TD3Obstacle policy:
    - For size-1 groups: use individual robot's action directly
    - For size-2/3 groups: average linear velocities of robots in the group
      to get coupled linear velocity, keep individual angular velocities
    
    This follows the same pattern as ShortHorizonOracle in coupled_action_oracle_eval.py.
    """
    
    def __init__(
        self,
        sim,                          # MARL_SIM_OBSTACLE instance
        policy,                       # TD3Obstacle policy (decentralized)
        groups: List[List[int]],
        horizon: int = 10,            # Number of steps to simulate forward
        n_rollouts_per_group: int = 1,
        device: torch.device = None,
    ):
        """
        Args:
            sim: MARL_SIM_OBSTACLE simulation environment
            policy: Trained TD3Obstacle policy (decentralized)
            groups: List of candidate groups
            horizon: Number of simulation steps for oracle evaluation
            n_rollouts_per_group: Number of rollouts to average for each group score
            device: Device for tensors
        """
        self.sim = sim
        self.policy = policy
        self.groups = groups
        self.horizon = horizon
        self.n_rollouts_per_group = n_rollouts_per_group
        self.device = device or torch.device("cpu")
        self.num_robots = sim.num_robots
        self.num_obstacles = sim.num_obstacles
    
    def get_action_for_group(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        group: List[int],
    ) -> List[List[float]]:
        """
        Get action for a specific group using the decentralized policy.
        
        For groups with size > 1, we compute the coupled linear velocity by
        taking the minimum of the linear velocities of all robots in the group.
        Each robot keeps its individual angular velocity.
        
        Args:
            robot_obs: Robot observations, shape (num_robots, state_dim).
            obstacle_obs: Obstacle observations, shape (num_obstacles, obs_dim).
            group: List of robot indices in the active group.
            
        Returns:
            Actions for all robots, shape (num_robots, 2).
            Inactive robots get [0, 0].
        """
        # Get actions from decentralized policy for all robots
        action, _ = self.policy.get_action(
            robot_obs, obstacle_obs, add_noise=False
        )
        # action is (num_robots, 2) with values in [-1, 1]
        
        group_size = len(group)
        
        if group_size == 1:
            # Size-1: Use individual robot's action directly
            robot_idx = group[0]
            # Scale linear velocity: [-1, 1] -> [0, 0.5] using (v + 1) / 4
            a_out = []
            for i in range(self.num_robots):
                if i == robot_idx:
                    scaled_lin_vel = (action[i][0] + 1) / 4  # [-1,1] -> [0,0.5]
                    a_out.append([scaled_lin_vel, action[i][1]])
                else:
                    a_out.append([0.0, 0.0])
            return a_out
        else:
            # Size-2/3: Compute coupled linear velocity using minimum
            # Get scaled linear velocities for robots in the group
            scaled_lin_vels = []
            for idx in group:
                scaled_lin_vel = (action[idx][0] + 1) / 4  # [-1,1] -> [0,0.5]
                scaled_lin_vels.append(scaled_lin_vel)
            
            # Use minimum linear velocity as the coupled velocity
            # This ensures safety - coupled robots move at the slowest robot's speed
            v_coupled = min(scaled_lin_vels)
            
            # Build output actions
            a_out = []
            for i in range(self.num_robots):
                if i in group:
                    # Use coupled linear velocity, individual angular velocity
                    a_out.append([v_coupled, action[i][1]])
                else:
                    a_out.append([0.0, 0.0])
            return a_out
    
    def compute_evasion_reward(
        self,
        group: List[int],
        initial_poses: List[List[float]],
        final_poses: List[List[float]],
        initial_obstacle_states: np.ndarray,
        final_obstacle_states: np.ndarray,
        robot_proximity_threshold: float = 1.25,
        obstacle_proximity_threshold: float = 1.25,
        robot_align_threshold: float = 0.7,
        obstacle_align_threshold: float = 0.7,
        return_details: bool = False,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Compute evasion reward for robots that rotate/move away from nearby entities.
        
        Rewards robots for:
        - Rotating away from nearby robots/obstacles (alignment improvement) - uses smaller threshold
        - Increasing clearance from nearby robots/obstacles (clearance improvement) - uses larger threshold
        
        NOTE: Uses proper clearance calculation:
        - Robot-robot clearance: center distance minus 2*robot_radius (0.4)
        - Robot-obstacle clearance: center distance minus (obstacle_radius+robot_radius) (0.9)

        Args:
            group: Robot indices in the group
            initial_poses: Per-robot poses at start [[x, y, theta], ...]
            final_poses: Per-robot poses at end [[x, y, theta], ...]
            initial_obstacle_states: Obstacle states at start (N_obs, 4) [x, y, vx, vy]
            final_obstacle_states: Obstacle states at end
            robot_proximity_threshold: Consider robots within this clearance for clearance reward
            obstacle_proximity_threshold: Consider obstacles within this clearance for clearance reward
            robot_align_threshold: Consider robots within this clearance for alignment reward (smaller)
            obstacle_align_threshold: Consider obstacles within this clearance for alignment reward (smaller)
            return_details: If True, return breakdown of evasion components
            
        Returns:
            evasion_score: Reward for evasive maneuvers (higher = better)
            details: Optional dict with breakdown (if return_details=True)
        """
        evasion_score = 0.0
        k_align = 5.0   # Weight for alignment improvement
        k_dist = 3.0    # Weight for clearance improvement
        
        # Geometry constants
        robot_radius = 0.2       # Robot radius from world yaml
        obstacle_radius = 0.7   # Obstacle radius from world yaml
        robot_collision_dist = 2 * robot_radius  # 0.4m center-to-center for collision
        obstacle_collision_dist = obstacle_radius + robot_radius  # 0.9m center-to-center for collision
        # Track detailed breakdown
        robot_align_total = 0.0
        robot_dist_total = 0.0
        obs_align_total = 0.0
        obs_dist_total = 0.0
        
        for i in group:
            # Robot i's initial and final state
            xi_init, yi_init, theta_i_init = initial_poses[i]
            xi_final, yi_final, theta_i_final = final_poses[i]
            
            # === Robot-Robot Evasion ===
            for j in range(self.num_robots):
                if i == j:
                    continue
                
                # Initial center-to-center distance and clearance to robot j
                xj_init, yj_init, _ = initial_poses[j]
                dx_init = xj_init - xi_init
                dy_init = yj_init - yi_init
                center_dist_init = np.sqrt(dx_init**2 + dy_init**2)
                # Clearance = center distance - 2*robot_radius
                clearance_init = center_dist_init - robot_collision_dist
                
                # Skip if outside both thresholds
                if clearance_init > robot_proximity_threshold:
                    continue
                
                # Angle from robot i to robot j (in world frame)
                angle_to_j_init = np.arctan2(dy_init, dx_init)
                
                # Alignment: cos(heading - angle_to_entity)
                # = 1 means pointing directly at entity, -1 means pointing away
                alignment_init = np.cos(theta_i_init - angle_to_j_init)
                
                # Final state
                xj_final, yj_final, _ = final_poses[j]
                dx_final = xj_final - xi_final
                dy_final = yj_final - yi_final
                center_dist_final = np.sqrt(dx_final**2 + dy_final**2)
                clearance_final = center_dist_final - robot_collision_dist
                angle_to_j_final = np.arctan2(dy_final, dx_final)
                alignment_final = np.cos(theta_i_final - angle_to_j_final)
                
                # Clearance improvement (uses robot_proximity_threshold)
                # Clip final clearance at threshold - no extra reward for moving beyond safe distance
                clearance_final_clipped = min(clearance_final, robot_proximity_threshold)
                clearance_improvement = clearance_final_clipped - clearance_init
                urgency_dist = max(0, robot_proximity_threshold - clearance_init) / robot_proximity_threshold
                robot_dist_contrib = urgency_dist * k_dist * clearance_improvement
                robot_dist_total += robot_dist_contrib
                evasion_score += robot_dist_contrib
                
                # Alignment improvement (uses smaller robot_align_threshold)
                if clearance_init <= robot_align_threshold:
                    alignment_improvement = alignment_init - alignment_final
                    urgency_align = max(0, robot_align_threshold - clearance_init) / robot_align_threshold
                    robot_align_contrib = urgency_align * k_align * alignment_improvement
                    robot_align_total += robot_align_contrib
                    evasion_score += robot_align_contrib
            
            # === Robot-Obstacle Evasion ===
            for obs_idx in range(self.num_obstacles):
                # Initial obstacle position
                ox_init = initial_obstacle_states[obs_idx, 0]
                oy_init = initial_obstacle_states[obs_idx, 1]
                
                dx_init = ox_init - xi_init
                dy_init = oy_init - yi_init
                center_dist_init = np.sqrt(dx_init**2 + dy_init**2)
                # Clearance = center distance - obstacle_collision_dist 
                clearance_init = center_dist_init - obstacle_collision_dist
                
                # Skip if outside both thresholds
                if clearance_init > obstacle_proximity_threshold:
                    continue
                
                angle_to_obs_init = np.arctan2(dy_init, dx_init)
                alignment_init = np.cos(theta_i_init - angle_to_obs_init)
                
                # Final obstacle position
                ox_final = final_obstacle_states[obs_idx, 0]
                oy_final = final_obstacle_states[obs_idx, 1]
                
                dx_final = ox_final - xi_final
                dy_final = oy_final - yi_final
                center_dist_final = np.sqrt(dx_final**2 + dy_final**2)
                clearance_final = center_dist_final - obstacle_collision_dist
                angle_to_obs_final = np.arctan2(dy_final, dx_final)
                alignment_final = np.cos(theta_i_final - angle_to_obs_final)
                
                # Clearance improvement (uses obstacle_proximity_threshold)
                # Clip final clearance at threshold - no extra reward for moving beyond safe distance
                clearance_final_clipped = min(clearance_final, obstacle_proximity_threshold)
                clearance_improvement = clearance_final_clipped - clearance_init
                urgency_dist = max(0, obstacle_proximity_threshold - clearance_init) / obstacle_proximity_threshold
                obs_dist_contrib = urgency_dist * k_dist * clearance_improvement
                obs_dist_total += obs_dist_contrib
                evasion_score += obs_dist_contrib
                
                # Alignment improvement (uses smaller obstacle_align_threshold)
                if clearance_init <= obstacle_align_threshold:
                    alignment_improvement = alignment_init - alignment_final
                    urgency_align = max(0, obstacle_align_threshold - clearance_init) / obstacle_align_threshold
                    obs_align_contrib = urgency_align * k_align * alignment_improvement
                    obs_align_total += obs_align_contrib
                    evasion_score += obs_align_contrib
        
        details = {
            'robot_align': robot_align_total,
            'robot_dist': robot_dist_total,
            'obs_align': obs_align_total,
            'obs_dist': obs_dist_total,
        }
        
        if return_details:
            return evasion_score, details
        return evasion_score, None
    
    def compute_stuckness_penalty(
        self,
        group: List[int],
        initial_poses: List[List[float]],
        final_poses: List[List[float]],
        min_displacement_threshold: float = 0.2,
        had_new_reach: bool = False,
        reached: Optional[List[bool]] = None,
    ) -> float:
        """
        Compute penalty for groups that result in very low movement (stuckness).
        
        This discourages the switcher from selecting groups that don't make progress.
        Skips already-reached robots (they may orbit near goal, that's OK).
        
        Args:
            group: Robot indices in the group
            initial_poses: Per-robot poses at start [[x, y, theta], ...]
            final_poses: Per-robot poses at end [[x, y, theta], ...]
            min_displacement_threshold: Minimum expected displacement over horizon
            had_new_reach: If True, don't penalize (a robot newly reached goal)
            reached: Per-robot reached flags (skip already-reached robots)
            
        Returns:
            stuckness_penalty: Negative value if group is stuck (lower = worse)
        """
        if had_new_reach:
            # Don't penalize if a new goal was reached during this rollout
            return 0.0
        
        k_stuck = 20.0
        
        # Filter out already-reached robots from stuckness check
        unreached_in_group = [
            i for i in group 
            if reached is None or not reached[i]
        ]
        
        if len(unreached_in_group) == 0:
            # All robots in group already reached — no stuckness penalty
            return 0.0
        
        # Compute average displacement of unreached robots in the group
        total_displacement = 0.0
        for i in unreached_in_group:
            xi_init, yi_init, _ = initial_poses[i]
            xi_final, yi_final, _ = final_poses[i]
            displacement = np.sqrt((xi_final - xi_init)**2 + (yi_final - yi_init)**2)
            total_displacement += displacement
        
        avg_displacement = total_displacement / len(unreached_in_group)
        
        # Penalize if below threshold
        if avg_displacement < min_displacement_threshold:
            return -k_stuck * (min_displacement_threshold - avg_displacement)
        
        return 0.0
    
    def compute_trajectory_score(
        self,
        group: List[int],
        initial_poses: List[List[float]],
        final_poses: List[List[float]],
        initial_distances: List[float],
        final_distances: List[float],
        initial_obstacle_states: np.ndarray,
        final_obstacle_states: np.ndarray,
        had_collision: bool,
        n_new_reached: int,
        n_already_reached_before: int,
        reached_before_rollout: List[bool],
        robot_proximity_threshold: float = 1.5,
        obstacle_proximity_threshold: float = 1.5,
        min_displacement_threshold: float = 0.2,
        return_breakdown: bool = False,
    ) -> float:
        """
        Compute trajectory-based score for an oracle rollout.
        
        Revised scoring for synchronized goal arrival:
        
        1. Collision penalty: -50 if any collision occurred
        2. New-reach bonus: k_reach / n_remaining per newly reached robot
           (more valuable when fewer robots remain unreached)
        3. Progress reward: Laggard-weighted — robots farther from goal get
           more reward per meter of progress. Already-reached robots excluded.
        4. Synchronization reward: Reduces variance of dist_to_goal across
           ALL robots — encourages balanced progress.
        5. Evasion reward: Unchanged from original.
        6. Stuckness penalty: Unchanged, but skips already-reached robots.
        
        Args:
            group: Robot indices in the group
            initial_poses: Per-robot poses at start [[x, y, theta], ...]
            final_poses: Per-robot poses at end [[x, y, theta], ...]
            initial_distances: Per-robot distances to goal at step 0
            final_distances: Per-robot distances to goal at final step
            initial_obstacle_states: Obstacle states at start
            final_obstacle_states: Obstacle states at end
            had_collision: Whether collision occurred during trajectory
            n_new_reached: Number of robots that newly got reached flag
            n_already_reached_before: Number of robots already reached before rollout
            reached_before_rollout: Per-robot reached flags before this rollout
            robot_proximity_threshold: Threshold for evasion reward (robots)
            obstacle_proximity_threshold: Threshold for evasion reward (obstacles)
            min_displacement_threshold: Minimum displacement to avoid stuckness penalty
            return_breakdown: If True, return (score, breakdown_dict) instead of just score
            
        Returns:
            score: Trajectory score (higher = better)
            If return_breakdown=True, returns (score, breakdown_dict).
        """
        N = self.num_robots
        k_reach = CONFIG.get("k_reach", 50.0)
        k_progress = CONFIG.get("k_progress", 3.0)
        k_sync = CONFIG.get("k_sync", 8.0)
        
        # Breakdown dict for debug
        breakdown = {
            "collision_penalty": 0.0,
            "reach_bonus": 0.0,
            "progress_reward": 0.0,
            "sync_reward": 0.0,
            "evasion_reward": 0.0,
            "stuckness_penalty": 0.0,
            "total": 0.0,
            "n_new_reached": n_new_reached,
            "n_already_reached_before": n_already_reached_before,
            "had_collision": had_collision,
        }
        
        # 1. Collision penalty
        if had_collision:
            breakdown["collision_penalty"] = -50.0
            breakdown["total"] = -50.0
            if return_breakdown:
                return -50.0, breakdown
            return -50.0
        
        score = 0.0
        
        # 2. New-reach bonus: k_reach / n_remaining per newly reached robot
        # The fewer robots remaining, the more valuable each new reach is.
        reach_bonus = 0.0
        if n_new_reached > 0:
            n_remaining_before = N - n_already_reached_before
            if n_remaining_before > 0:
                for r in range(n_new_reached):
                    remaining_at_time = n_remaining_before - r
                    if remaining_at_time > 0:
                        reach_bonus += k_reach / remaining_at_time
        score += reach_bonus
        breakdown["reach_bonus"] = reach_bonus
        
        # 3. Laggard-weighted progress reward
        unreached_indices = [
            i for i in range(N) if not reached_before_rollout[i]
        ]
        progress_reward = 0.0
        if len(unreached_indices) > 0:
            unreached_dists = [initial_distances[i] for i in unreached_indices]
            mean_dist = np.mean(unreached_dists) if len(unreached_dists) > 0 else 1.0
            mean_dist = max(mean_dist, 0.1)
            
            for i in group:
                if reached_before_rollout[i]:
                    continue
                progress = initial_distances[i] - final_distances[i]
                laggard_weight = initial_distances[i] / mean_dist
                progress_reward += k_progress * progress * laggard_weight
        score += progress_reward
        breakdown["progress_reward"] = progress_reward
        
        # 4. Synchronization reward
        # Use ALL robots (reached robots have dist ≈ 0, naturally "synchronized").
        # If we used only unreached robots, moving one robot toward goal would
        # increase variance among the unreached subset, penalizing good progress.
        sync_reward = 0.0
        if N >= 2:
            initial_all_dists = np.array(initial_distances)
            final_all_dists = np.array(final_distances)
            var_before = np.var(initial_all_dists)
            var_after = np.var(final_all_dists)
            sync_reward = k_sync * (var_before - var_after)
        score += sync_reward
        breakdown["sync_reward"] = sync_reward
        
        # 5. Evasion reward
        evasion_reward, evasion_details = self.compute_evasion_reward(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            initial_obstacle_states=initial_obstacle_states,
            final_obstacle_states=final_obstacle_states,
            robot_proximity_threshold=robot_proximity_threshold,
            obstacle_proximity_threshold=obstacle_proximity_threshold,
            return_details=return_breakdown,
        )
        score += evasion_reward
        breakdown["evasion_reward"] = evasion_reward
        if evasion_details:
            breakdown["evasion_details"] = evasion_details
        
        # 6. Stuckness penalty
        stuckness_penalty = self.compute_stuckness_penalty(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            min_displacement_threshold=min_displacement_threshold,
            had_new_reach=(n_new_reached > 0),
            reached=reached_before_rollout,
        )
        score += stuckness_penalty
        breakdown["stuckness_penalty"] = stuckness_penalty
        
        breakdown["total"] = score
        
        if return_breakdown:
            return score, breakdown
        return score
    
    def _evaluate_group_once(
        self,
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
        reached: Optional[List[bool]] = None,
        return_breakdown: bool = False,
    ) -> Tuple[float, bool]:
        """
        Evaluate a group by simulating forward H steps (single rollout).
        
        Uses trajectory-based scoring that evaluates the entire rollout:
        - Collision penalty if any collision occurred
        - New-reach bonus for robots newly reaching goal
        - Laggard-weighted progress reward
        - Synchronization reward (variance reduction)
        - Evasion reward for rotating/moving away from nearby entities
        - Stuckness penalty for groups with very low displacement
        
        Tracks sticky reached[] flags per rollout to detect new reaches.
        
        Args:
            group: Robot indices in the group.
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after rollout.
            reached: Episode-level sticky reached flags per robot (None = all False).
            return_breakdown: If True, return (score, had_collision, breakdown_dict).
            
        Returns:
            Tuple of (trajectory_score, had_collision).
            If return_breakdown, returns (trajectory_score, had_collision, breakdown_dict).
        """
        N = self.num_robots
        goal_threshold = CONFIG.get("goal_reach_threshold", 0.3)
        
        if reached is None:
            reached = [False] * N
        reached_before = list(reached)  # snapshot for scoring
        n_already_reached = sum(reached_before)
        
        had_collision = False
        n_new_reached = 0
        rollout_reached = list(reached_before)  # sticky copy for this rollout
        
        # Store initial state for scoring
        initial_poses = [p.copy() for p in poses]
        initial_distances = distance.copy()
        initial_obstacle_states = obstacle_states.copy()
        
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
            # Prepare state using the policy's prepare_state method
            robot_state, _ = self.policy.prepare_state(
                curr_poses, curr_distance, curr_cos, curr_sin, 
                curr_collision, curr_action, curr_goal_positions
            )
            
            # Get action for this group
            a_in = self.get_action_for_group(
                np.array(robot_state),
                curr_obstacle_states,
                group,
            )
            
            # Step simulation
            (
                curr_poses, curr_distance, curr_cos, curr_sin,
                curr_collision, curr_goal, curr_action, reward,
                _, curr_goal_positions, curr_obstacle_states
            ) = self.sim.step(a_in, None, None)
            
            # Check for NEW goal reaches (sticky flag)
            # A robot is "reached" if distance < threshold and not already flagged
            for i in range(N):
                if not rollout_reached[i] and curr_distance[i] < goal_threshold:
                    rollout_reached[i] = True
                    n_new_reached += 1
            
            # Check for collision - end rollout early if collision
            if any(curr_collision[i] for i in group):
                had_collision = True
                break
            
            # Check for out of bounds
            if outside_of_bounds(curr_poses, self.sim):
                had_collision = True
                break
        
        # Final state after trajectory
        final_poses = [p.copy() for p in curr_poses]
        final_distances = curr_distance.copy()
        final_obstacle_states = curr_obstacle_states.copy()
        
        # Compute trajectory score using start and end states
        score_result = self.compute_trajectory_score(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            initial_distances=initial_distances,
            final_distances=final_distances,
            initial_obstacle_states=initial_obstacle_states,
            final_obstacle_states=final_obstacle_states,
            had_collision=had_collision,
            n_new_reached=n_new_reached,
            n_already_reached_before=n_already_reached,
            reached_before_rollout=reached_before,
            robot_proximity_threshold=1.5,
            obstacle_proximity_threshold=self.sim.obstacle_proximity_threshold,
            min_displacement_threshold=0.2,
            return_breakdown=return_breakdown,
        )
        
        # Restore simulation to original state
        snapshot.restore_to_sim(self.sim)
        
        if return_breakdown:
            trajectory_score, breakdown = score_result
            # Add distance info to breakdown for debug
            breakdown["initial_distances"] = initial_distances
            breakdown["final_distances"] = final_distances
            return trajectory_score, had_collision, breakdown
        else:
            trajectory_score = score_result
            return trajectory_score, had_collision
    
    def _evaluate_group(
        self,
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
        reached: Optional[List[bool]] = None,
        return_breakdown: bool = False,
    ) -> float:
        """
        Evaluate a group by averaging over n_rollouts.
        
        Args:
            group: List of robot indices in the group
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after each rollout.
            reached: Episode-level sticky reached flags per robot.
            return_breakdown: If True, return (score, breakdown_dict).
            
        Returns:
            score: Average cumulative reward across rollouts (higher = better)
            If return_breakdown, returns (score, breakdown_dict) from the last rollout.
        """
        total_reward = 0.0
        last_breakdown = None
        
        for rollout_idx in range(self.n_rollouts_per_group):
            if return_breakdown:
                reward, _, breakdown = self._evaluate_group_once(
                    group, poses, distance, cos, sin, collision, action,
                    goal_positions, obstacle_states, snapshot,
                    reached=reached,
                    return_breakdown=True,
                )
                last_breakdown = breakdown
            else:
                reward, _ = self._evaluate_group_once(
                    group, poses, distance, cos, sin, collision, action,
                    goal_positions, obstacle_states, snapshot,
                    reached=reached,
                )
            total_reward += reward
        
        avg_score = total_reward / self.n_rollouts_per_group
        if return_breakdown:
            return avg_score, last_breakdown
        return avg_score
    
    def _get_embeddings_and_attention(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get robot embeddings and attention weights from the decentralized policy.
        
        Uses the TD3Obstacle actor's attention module to get embeddings and
        attention weights. The embeddings are the same as used in the policy.
        
        Args:
            robot_obs: Robot observations, shape (num_robots, state_dim)
            obstacle_obs: Obstacle observations, shape (num_obstacles, obs_dim)
            
        Returns:
            h: Per-robot embeddings, Tensor[N, embed_dim*2]
            attn_rr: Robot-robot attention weights, Tensor[N, N]
            attn_ro: Robot-obstacle attention weights, Tensor[N, N_obs]
        """
        robot_tensor = torch.tensor(robot_obs, dtype=torch.float32, device=self.device)
        obstacle_tensor = torch.tensor(obstacle_obs, dtype=torch.float32, device=self.device)
        
        # Add batch dimension
        robot_tensor = robot_tensor.unsqueeze(0)  # (1, N, state_dim)
        obstacle_tensor = obstacle_tensor.unsqueeze(0)  # (1, N_obs, obs_dim)
        
        with torch.no_grad():
            # Get embeddings and attention from the actor's attention module
            (
                H,  # Per-robot embeddings: (B*N, embed_dim*2)
                hard_logits_rr,
                hard_logits_ro,
                dist_rr,
                dist_ro,
                mean_entropy,
                hard_weights_rr,  # (B, N, N)
                hard_weights_ro,  # (B, N, N_obs)
                combined_weights,
            ) = self.policy.actor.attention(robot_tensor, obstacle_tensor)
        
        # Reshape H from (B*N, embed_dim*2) to (B, N, embed_dim*2), then remove batch dim
        batch_size = robot_tensor.shape[0]
        n_robots = robot_tensor.shape[1]
        embed_dim_2 = H.shape[-1]  # embed_dim * 2
        
        h = H.view(batch_size, n_robots, embed_dim_2).squeeze(0)  # (N, embed_dim*2)
        attn_rr = hard_weights_rr.squeeze(0)  # (N, N)
        attn_ro = hard_weights_ro.squeeze(0)  # (N, N_obs)
        
        return h, attn_rr, attn_ro
    
    def _get_extra_features(
        self,
        poses: List[List[float]],
        distance: List[float],
        goal_positions: List[List[float]],
        reached: Optional[List[bool]] = None,
        step_in_episode: int = 0,
        max_steps: int = 400,
    ) -> Dict[str, torch.Tensor]:
        """
        Get extra per-robot features from environment state.
        
        Args:
            poses: Per-robot poses [[x, y, theta], ...]
            distance: Per-robot distances to goal
            goal_positions: Per-robot goals [[gx, gy], ...]
            reached: Per-robot sticky reached flags
            step_in_episode: Current step in the episode
            max_steps: Maximum steps per episode
            
        Returns:
            extra: Dict with per-robot feature tensors.
        """
        N = self.num_robots
        
        # Distance to goal
        dist_to_goal = torch.tensor(distance, dtype=torch.float32)
        
        # Minimum clearance to obstacles
        clearances = []
        for i in range(N):
            min_clearance = self.sim.get_min_obstacle_clearance(i)
            clearances.append(min_clearance)
        clearance = torch.tensor(clearances, dtype=torch.float32)
        
        # Reached flags (broadcast per robot)
        if reached is None:
            reached = [False] * N
        reached_float = torch.tensor(
            [1.0 if r else 0.0 for r in reached], dtype=torch.float32
        )
        
        # Fraction of all robots that have reached (same value for all robots,
        # broadcast to per-robot tensor so GroupFeatureBuilder can aggregate)
        frac_reached_global_val = sum(reached) / N
        frac_reached_global = torch.full((N,), frac_reached_global_val, dtype=torch.float32)
        
        # Max dist_to_goal (per-robot tensor, same value for all — the max across all robots)
        # This gives context about the worst-case robot
        max_dist_val = max(distance) if distance else 0.0
        max_dist_to_goal = torch.full((N,), max_dist_val, dtype=torch.float32)
        
        # Variance of dist_to_goal across unreached robots
        unreached_dists = [distance[i] for i in range(N) if not reached[i]]
        if len(unreached_dists) >= 2:
            var_dist_val = float(np.var(unreached_dists))
        else:
            var_dist_val = 0.0
        var_dist_to_goal = torch.full((N,), var_dist_val, dtype=torch.float32)
        
        # Steps elapsed fraction (same for all robots)
        steps_frac = step_in_episode / max(max_steps, 1)
        steps_elapsed_frac = torch.full((N,), steps_frac, dtype=torch.float32)
        
        return {
            "dist_to_goal": dist_to_goal,
            "clearance": clearance,
            "reached": reached_float,
            "frac_reached_global": frac_reached_global,
            "max_dist_to_goal": max_dist_to_goal,
            "var_dist_to_goal": var_dist_to_goal,
            "steps_elapsed_frac": steps_elapsed_frac,
        }
    
    def collect_sample(
        self,
        poses: List[List[float]],
        distance: List[float],
        cos: List[float],
        sin: List[float],
        collision: List[bool],
        action: List[List[float]],
        goal_positions: List[List[float]],
        obstacle_states: np.ndarray,
        scenario_id: Optional[int] = None,
        reached: Optional[List[bool]] = None,
        step_in_episode: int = 0,
        max_steps: int = 400,
    ) -> Dict:
        """
        Collect one sample of oracle data at the current simulation state.
        
        1. Get robot embeddings and attention from policy
        2. For each group, run rollouts and compute average score
        3. Return sample dict with new extra features
        
        Args:
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state from sim.step() or sim.reset()
            scenario_id: Optional identifier for this sample
            reached: Episode-level sticky reached flags per robot
            step_in_episode: Current step in the episode
            max_steps: Maximum steps per episode
            
        Returns:
            Sample dictionary compatible with train_switcher.py
        """
        debug = CONFIG.get("debug_mode", False)
        
        # Take snapshot before any rollouts
        snapshot = SimulationSnapshot.from_sim(self.sim)
        
        # Prepare robot observations using the policy's prepare_state method
        robot_state, _ = self.policy.prepare_state(
            poses, distance, cos, sin, collision, action, goal_positions
        )
        robot_obs = np.array(robot_state)
        
        # Get embeddings and attention
        h, attn_rr, attn_ro = self._get_embeddings_and_attention(robot_obs, obstacle_states)
        
        # Get extra features (CPU, fast) — now includes reached, sync features
        extra = self._get_extra_features(
            poses, distance, goal_positions,
            reached=reached,
            step_in_episode=step_in_episode,
            max_steps=max_steps,
        )
        
        # Evaluate each group with rollouts (one at a time)
        group_scores = []
        all_breakdowns = [] if debug else None
        
        if debug:
            N = self.num_robots
            reached_str = "".join(["R" if (reached and reached[i]) else "." for i in range(N)])
            print(f"\n{'='*80}")
            print(f"  SAMPLE #{scenario_id}  |  ep_step={step_in_episode}  |  reached=[{reached_str}] ({sum(reached) if reached else 0}/{N})")
            print(f"  dist_to_goal: {['%.2f' % d for d in distance]}")
            print(f"{'='*80}")
        
        for g_idx, group in enumerate(self.groups):
            if debug:
                score, breakdown = self._evaluate_group(
                    group, poses, distance, cos, sin, collision, action,
                    goal_positions, obstacle_states, snapshot,
                    reached=reached,
                    return_breakdown=True,
                )
                all_breakdowns.append(breakdown)
            else:
                score = self._evaluate_group(
                    group, poses, distance, cos, sin, collision, action,
                    goal_positions, obstacle_states, snapshot,
                    reached=reached,
                )
            group_scores.append(score)
        
        group_scores_tensor = torch.tensor(group_scores, dtype=torch.float32)
        
        # === DEBUG OUTPUT: per-group score breakdown ===
        if debug and all_breakdowns:
            # Sort groups by score (descending) for display
            ranked = sorted(range(len(self.groups)), key=lambda i: group_scores[i], reverse=True)
            
            # Print header
            print(f"\n  {'Rank':>4}  {'Group':<14}  {'Total':>7}  {'Reach':>7}  {'Progr':>7}  "
                  f"{'Sync':>7}  {'Evasn':>7}  {'Stuck':>7}  {'Colsn':>7}  {'NewR':>4}")
            print(f"  {'-'*4}  {'-'*14}  {'-'*7}  {'-'*7}  {'-'*7}  "
                  f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*4}")
            
            # Show top 10 and bottom 5
            display_indices = ranked[:10] + ranked[-5:] if len(ranked) > 15 else ranked
            shown_separator = False
            for rank, g_idx in enumerate(ranked):
                if g_idx not in display_indices:
                    if not shown_separator:
                        print(f"  {'...':>4}  {'(... %d more ...)' % (len(ranked) - 15):<14}")
                        shown_separator = True
                    continue
                bd = all_breakdowns[g_idx]
                grp_str = str(self.groups[g_idx])
                print(f"  {rank+1:>4}  {grp_str:<14}  {bd['total']:>+7.2f}  {bd['reach_bonus']:>+7.2f}  "
                      f"{bd['progress_reward']:>+7.2f}  {bd['sync_reward']:>+7.2f}  "
                      f"{bd['evasion_reward']:>+7.2f}  {bd['stuckness_penalty']:>+7.2f}  "
                      f"{bd['collision_penalty']:>+7.2f}  {bd['n_new_reached']:>4}")
            
            # Score distribution summary
            scores_arr = np.array(group_scores)
            print(f"\n  Score stats: min={scores_arr.min():.2f}  max={scores_arr.max():.2f}  "
                  f"mean={scores_arr.mean():.2f}  std={scores_arr.std():.2f}")
            
            # Per-term magnitude summary (absolute mean across all groups)
            terms = ["reach_bonus", "progress_reward", "sync_reward", "evasion_reward", "stuckness_penalty", "collision_penalty"]
            print(f"  Term magnitudes (mean |value| across all groups):")
            for term in terms:
                vals = [abs(bd[term]) for bd in all_breakdowns]
                nonzero = [v for v in vals if v > 0]
                mean_abs = np.mean(vals) if vals else 0
                n_active = len(nonzero)
                print(f"    {term:<20s}: mean|val|={mean_abs:>7.3f}  active_in={n_active:>3}/{len(self.groups)} groups")
            print()
        
        return {
            "h": h.cpu(),
            "groups": self.groups,
            "group_scores": group_scores_tensor,
            "attn_rr": attn_rr.cpu(),
            "attn_ro": attn_ro.cpu(),
            "extra": {k: v.cpu() for k, v in extra.items()},
            "metadata": {
                "scenario_id": scenario_id,
                "step_in_episode": step_in_episode,
                "n_reached": sum(reached) if reached else 0,
            },
        }
    
    def collect_dataset(
        self,
        n_samples: int,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Collect a full dataset of oracle samples by running episodes.
        
        Episode structure (all-reach reset):
        - Maintain sticky reached[i] flags per robot.
        - When a robot's distance < goal_threshold, mark reached[i] = True.
        - Robot keeps moving normally (no parking, no new goal).
        - Reset ALL robots when all reached, or max steps exceeded.
        
        Args:
            n_samples: Number of samples to collect
            save_path: Path to save the dataset (optional)
            verbose: Print progress bar
            
        Returns:
            data: Dataset dictionary
        """
        debug = CONFIG.get("debug_mode", False)
        samples = []
        
        pbar = tqdm(range(n_samples), desc="Collecting oracle data") if (verbose and not debug) else range(n_samples)
        
        # Reset environment to start
        (
            poses, distance, cos, sin, collision, goals,
            action, reward, positions, goal_positions, obstacle_states
        ) = self.sim.reset()
        
        step_in_episode = 0
        max_steps = CONFIG.get("max_steps_per_episode", 400)
        goal_threshold = CONFIG.get("goal_reach_threshold", 0.3)
        N = self.num_robots
        
        # Episode-level sticky reached flags
        reached = [False] * N
        
        # Statistics
        episode_count = 0
        all_reached_count = 0  # Episodes where all robots reached
        
        for i in (range(n_samples) if debug else pbar):
            if debug:
                print(f"\n{'#'*80}")
                print(f"  [DEBUG] Sample {i+1}/{n_samples}  |  Episode {episode_count}  |  Step {step_in_episode}")
                print(f"  Poses: {['(%.2f,%.2f)' % (p[0],p[1]) for p in poses]}")
                print(f"  Goals: {['(%.2f,%.2f)' % (g[0],g[1]) for g in goal_positions]}")
                print(f"  Distances: {['%.3f' % d for d in distance]}")
                reached_str = "".join(["R" if reached[r] else "." for r in range(N)])
                print(f"  Reached: [{reached_str}]  ({sum(reached)}/{N})")
                print(f"  Collision: {collision}")
                print(f"{'#'*80}")
            
            # Collect sample at current state (pass reached state)
            sample = self.collect_sample(
                poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states,
                scenario_id=i,
                reached=reached,
                step_in_episode=step_in_episode,
                max_steps=max_steps,
            )
            samples.append(sample)
            
            # Debug: pause for user inspection
            if debug:
                try:
                    user_input = input("[DEBUG] Press Enter to continue (q to quit debug, s to skip pauses)... ")
                    if user_input.strip().lower() == "q":
                        print("[DEBUG] Quitting debug mode, collecting remaining samples silently...")
                        CONFIG["debug_mode"] = False
                        debug = False
                    elif user_input.strip().lower() == "s":
                        print("[DEBUG] Skipping pauses, continuing with debug printing...")
                        # Keep debug prints but stop pausing — we use a local flag
                        # Simpler: just disable debug entirely
                        pass  # Continue normally, user can Ctrl+C if needed
                except KeyboardInterrupt:
                    print("\n[DEBUG] Interrupted. Returning collected samples so far.")
                    break
            
            # =================================================================
            # PHASE 2: Advance simulation with a selected group
            #          Run H steps (same horizon as oracle) to get a
            #          meaningfully different configuration for next sample.
            #
            #   Selection strategies:
            #     "random" — uniform random from all groups
            #     "top_k"  — random sample from the top-k oracle-scored groups
            # =================================================================
            phase2_mode = CONFIG.get("phase2_selection", "random")
            phase2_top_k = CONFIG.get("phase2_top_k", 5)
            
            if phase2_mode == "top_k":
                # Use Phase 1 oracle scores to pick from top-k groups
                scores = sample["group_scores"]  # Tensor[n_groups]
                k = min(phase2_top_k, len(self.groups))
                _, top_indices = torch.topk(scores, k)
                chosen_idx = top_indices[random.randint(0, k - 1)].item()
                selected_group = self.groups[chosen_idx]
            else:
                # Original: uniform random
                selected_group = random.choice(self.groups)
            
            phase2_horizon = self.horizon  # Same as oracle horizon (e.g. 10)
            
            # Capture pre-phase2 state for debug comparison
            if debug:
                pre_phase2_poses = [p.copy() for p in poses]
                pre_phase2_distances = list(distance)
                pre_phase2_obstacle_states = obstacle_states.copy()
                print(f"\n  {'='*72}")
                print(f"  PHASE 2: Advance Simulation  |  {phase2_horizon} sim.steps  |  mode={phase2_mode}")
                print(f"           group={selected_group} (size {len(selected_group)})")
                if phase2_mode == "top_k":
                    score_of_chosen = scores[chosen_idx].item()
                    top_k_scores = scores[top_indices].tolist()
                    print(f"           top-{k} scores: {['%.2f' % s for s in top_k_scores]}  |  chosen score: {score_of_chosen:.2f}")
                print(f"  {'='*72}")
            
            phase2_early_stop = False
            phase2_steps_done = 0
            
            for ph2_step in range(phase2_horizon):
                # Re-query policy at each step with current observations
                robot_state, _ = self.policy.prepare_state(
                    poses, distance, cos, sin, collision, action, goal_positions
                )
                robot_obs = np.array(robot_state)
                scaled_action = self.get_action_for_group(
                    robot_obs, obstacle_states, selected_group
                )
                
                if debug and ph2_step == 0:
                    # Print actions on first step to verify non-zero velocities
                    print(f"  Step {ph2_step+1}/{phase2_horizon} actions:")
                    for r_idx in range(N):
                        marker = " <-- ACTIVE" if r_idx in selected_group else ""
                        print(f"    Robot {r_idx}: lin_vel={scaled_action[r_idx][0]:.4f}, "
                              f"ang_vel={scaled_action[r_idx][1]:.4f}{marker}")
                
                # Step simulation
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states
                ) = self.sim.step(scaled_action, None, None)
                
                step_in_episode += 1
                phase2_steps_done += 1
                
                # Update sticky reached flags within Phase 2
                for r_idx in range(N):
                    if not reached[r_idx] and distance[r_idx] < goal_threshold:
                        reached[r_idx] = True
                        if debug:
                            print(f"    [Phase2 step {ph2_step+1}] *** Robot {r_idx} reached goal! dist={distance[r_idx]:.3f} ***")
                
                # Check for early termination conditions
                if any(collision):
                    phase2_early_stop = True
                    if debug:
                        collided = [r for r in range(N) if collision[r]]
                        print(f"    [Phase2 step {ph2_step+1}] COLLISION at robots {collided} — stopping Phase 2 early")
                    break
                if outside_of_bounds(poses, self.sim):
                    phase2_early_stop = True
                    if debug:
                        print(f"    [Phase2 step {ph2_step+1}] OUT OF BOUNDS — stopping Phase 2 early")
                    break
                if all(reached):
                    phase2_early_stop = True
                    if debug:
                        print(f"    [Phase2 step {ph2_step+1}] ALL ROBOTS REACHED — stopping Phase 2 early")
                    break
                if step_in_episode >= max_steps:
                    phase2_early_stop = True
                    if debug:
                        print(f"    [Phase2 step {ph2_step+1}] MAX STEPS ({max_steps}) — stopping Phase 2 early")
                    break
            
            # Debug: print Phase 2 summary (before vs after all H steps)
            if debug:
                print(f"\n  Phase 2 summary: {phase2_steps_done}/{phase2_horizon} steps executed"
                      f"{'  (early stop)' if phase2_early_stop else ''}")
                print(f"  Robot state changes (over {phase2_steps_done} steps):")
                print(f"  {'':>1}{'Robot':>5}  {'Before (x,y)':<18}  {'After (x,y)':<18}  "
                      f"{'displacement':>12}  {'d_before':>8}  {'d_after':>8}  {'Δd':>7}")
                for r_idx in range(N):
                    bx, by = pre_phase2_poses[r_idx][0], pre_phase2_poses[r_idx][1]
                    ax, ay = poses[r_idx][0], poses[r_idx][1]
                    disp = np.sqrt((ax - bx)**2 + (ay - by)**2)
                    d_before = pre_phase2_distances[r_idx]
                    d_after = distance[r_idx]
                    dd = d_after - d_before
                    active = "*" if r_idx in selected_group else " "
                    reached_mark = " R" if reached[r_idx] else ""
                    print(f"  {active}{r_idx:>4}  ({bx:>7.3f},{by:>7.3f})  "
                          f"({ax:>7.3f},{ay:>7.3f})  "
                          f"{disp:>12.4f}  "
                          f"{d_before:>8.4f}  {d_after:>8.4f}  {dd:>+7.4f}{reached_mark}")
                
                reached_str = "".join(["R" if reached[r] else "." for r in range(N)])
                print(f"  Reached after Phase 2: [{reached_str}] ({sum(reached)}/{N})")
                print(f"  Collision: {collision}")
                print(f"  Episode step now: {step_in_episode}")
                print(f"  {'='*72}\n")
            if debug:
                user_input = input("[DEBUG] Press Enter to continue (q to quit debug, s to skip pauses)... ")

            # Check for episode reset conditions
            all_robots_reached = all(reached)
            should_reset = (
                any(collision) or 
                step_in_episode >= max_steps or
                outside_of_bounds(poses, self.sim) or
                all_robots_reached
            )
            
            if should_reset:
                if debug:
                    reason = []
                    if any(collision):
                        reason.append("COLLISION")
                    if step_in_episode >= max_steps:
                        reason.append("MAX_STEPS")
                    if outside_of_bounds(poses, self.sim):
                        reason.append("OUT_OF_BOUNDS")
                    if all_robots_reached:
                        reason.append("ALL_REACHED ✓")
                    print(f"\n  [DEBUG] === EPISODE RESET === reason: {', '.join(reason)} ===")
                
                if all_robots_reached:
                    all_reached_count += 1
                episode_count += 1
                
                # Full reset: all robots get new random positions and goals
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states
                ) = self.sim.reset(random_obstacles=True)
                step_in_episode = 0
                reached = [False] * N  # Reset all reached flags
            
            # Progress reporting
            if verbose and isinstance(pbar, tqdm):
                n_reached = sum(reached)
                pbar.set_postfix({
                    "ep_step": step_in_episode,
                    "reached": f"{n_reached}/{N}",
                    "ep": episode_count,
                })
        
        if verbose:
            print(f"\nEpisodes: {episode_count} total, "
                  f"{all_reached_count} all-reached ({all_reached_count/max(episode_count,1)*100:.1f}%)")
        
        data = {
            "samples": samples,
            "config": {
                "n_samples": n_samples,
                "embed_dim": CONFIG["embed_dim"],
                "n_robots": self.num_robots,
                "n_obstacles": self.num_obstacles,
                "n_groups": len(self.groups),
                "groups": self.groups,
                "horizon": self.horizon,
                "n_rollouts_per_group": self.n_rollouts_per_group,
                "collection_method": "simulation_rollout_singleprocess_allreach",
                "scoring": "sync_newreach_laggard",
                "k_reach": CONFIG.get("k_reach", 50.0),
                "k_progress": CONFIG.get("k_progress", 3.0),
                "k_sync": CONFIG.get("k_sync", 8.0),
                "goal_reach_threshold": goal_threshold,
                "max_steps_per_episode": max_steps,
                "episodes_total": episode_count,
                "episodes_all_reached": all_reached_count,
                "extra_features": [
                    "dist_to_goal", "clearance", "reached",
                    "frac_reached_global", "max_dist_to_goal",
                    "var_dist_to_goal", "steps_elapsed_frac",
                ],
                "timestamp": datetime.now().isoformat(),
            },
        }
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, save_path)
            if verbose:
                print(f"Saved dataset to {save_path}")
        
        return data


def main():
    """Main function to collect oracle data."""
    from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
    from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
    
    config = CONFIG
    debug = config.get("debug_mode", False)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if debug:
        print("\n" + "!" * 70)
        print("!  [DEBUG MODE] Simulation rendering ON, keyboard pause enabled")
        print("!  Press Enter after each sample to advance, 'q' to exit debug")
        print("!" * 70 + "\n")
    
    print("=" * 70)
    print("Oracle Data Collection for Group Switcher (All-Reach Synchronized)")
    print("=" * 70)
    print(f"Output path: {config['output_path']}")
    n_samples = config["n_samples"]
    print(f"Number of samples: {n_samples}")
    print(f"Number of robots: {config['n_robots']}")
    print(f"Embedding dimension: {config['embed_dim']}")
    print(f"Oracle horizon: {config['oracle_horizon']} steps")
    print(f"Rollouts per group: {config['n_rollouts_per_group']}")
    print(f"Goal reach threshold: {config.get('goal_reach_threshold', 0.3)}")
    print(f"Scoring: new-reach bonus (k={config.get('k_reach', 50.0)}), "
          f"sync reward (k={config.get('k_sync', 8.0)}), "
          f"laggard progress (k={config.get('k_progress', 3.0)})")
    print(f"Episode reset: ALL robots reached OR max {config['max_steps_per_episode']} steps")
    phase2_mode = config.get("phase2_selection", "random")
    if phase2_mode == "top_k":
        print(f"Phase 2 selection: top-{config.get('phase2_top_k', 5)} (sample from top-k oracle-scored groups)")
    else:
        print(f"Phase 2 selection: random (uniform random group)")
    if debug:
        print(f"Debug mode: ON — rendering enabled, keyboard pauses active")
    print("=" * 70 + "\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create simulation environment
    # In debug mode, disable_plotting=False so the simulation renders
    logger.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=not debug,  # False when debug (render), True otherwise
        reward_phase=5,
        per_robot_goal_reset=False,  # All-reach: don't reset individual goals
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )
    logger.info(f"Environment: {sim.num_robots} robots, {sim.num_obstacles} obstacles")
    
    # Load decentralized policy (TD3Obstacle)
    logger.info("Loading decentralized policy (TD3Obstacle)...")
    policy = TD3Obstacle(
        state_dim=config["state_dim"],
        action_dim=2,
        max_action=1.0,
        device=device,
        num_robots=config["n_robots"],
        num_obstacles=config["n_obstacles"],
        obstacle_state_dim=config["obstacle_state_dim"],
        load_model=True,
        model_name=config["decentralized_model_name"],
        load_model_name=config["decentralized_model_name"],
        load_directory=Path(config["decentralized_model_directory"]),
        save_directory=Path(config["decentralized_model_directory"]),
    )
    logger.info("Loaded decentralized policy successfully")
    
    # Generate candidate groups
    candidate_groups = generate_candidate_groups(
        num_robots=config["n_robots"],
        include_size_1=config["include_size_1"],
        include_size_2=config["include_size_2"],
        include_size_3=config["include_size_3"],
    )
    
    logger.info(f"Candidate groups: {len(candidate_groups)} total")
    logger.info(f"  Size-1: {sum(1 for g in candidate_groups if len(g) == 1)}")
    logger.info(f"  Size-2: {sum(1 for g in candidate_groups if len(g) == 2)}")
    logger.info(f"  Size-3: {sum(1 for g in candidate_groups if len(g) == 3)}")
    
    # Create oracle data collector
    collector = OracleDataCollector(
        sim=sim,
        policy=policy,
        groups=candidate_groups,
        horizon=config["oracle_horizon"],
        n_rollouts_per_group=config["n_rollouts_per_group"],
        device=device,
    )
    
    # Collect data
    data = collector.collect_dataset(
        n_samples=n_samples,
        save_path=None,  # We'll save below
        verbose=True,
    )
    
    # Save
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    
    n_robots = config["n_robots"]
    embed_dim = config["embed_dim"]
    n_groups = data["config"]["n_groups"]
    
    print(f"\nSaved {len(data['samples'])} samples to {output_path}")
    print(f"\nData format:")
    print(f"  samples[i]['h']: Tensor[{n_robots}, {embed_dim * 2}]  (per-robot embeddings)")
    print(f"  samples[i]['groups']: List of {n_groups} groups")
    print(f"  samples[i]['group_scores']: Tensor[{n_groups}]  (oracle reward scores)")
    print(f"  samples[i]['attn_rr']: Tensor[{n_robots}, {n_robots}]")
    print(f"  samples[i]['attn_ro']: Tensor[{n_robots}, N_obs]")
    print(f"  samples[i]['extra']:")
    print(f"    'dist_to_goal': Tensor[{n_robots}]")
    print(f"    'clearance': Tensor[{n_robots}]")
    print(f"    'reached': Tensor[{n_robots}]  (sticky binary flag)")
    print(f"    'frac_reached_global': Tensor[{n_robots}]  (broadcast scalar)")
    print(f"    'max_dist_to_goal': Tensor[{n_robots}]  (broadcast scalar)")
    print(f"    'var_dist_to_goal': Tensor[{n_robots}]  (broadcast scalar)")
    print(f"    'steps_elapsed_frac': Tensor[{n_robots}]  (broadcast scalar)")
    print(f"\n  Scoring: new-reach bonus + laggard-weighted progress + sync reward")
    print(f"  Episode reset: all robots reached OR max {config['max_steps_per_episode']} steps")
    print(f"  Episodes: {data['config'].get('episodes_total', '?')} total, "
          f"{data['config'].get('episodes_all_reached', '?')} all-reached")
    
    print(f"\nTo train the switcher:")
    print(f"  python -m robot_nav.models.MARL.switcher.train_switcher")


if __name__ == "__main__":
    main()
