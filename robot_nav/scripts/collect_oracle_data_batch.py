"""
Oracle Data Collection Script for Group Switcher Training.
(Batch-optimized single-process GPU version)

This script collects training data for the GroupSwitcher by running
oracle evaluations (simulation rollouts) for different group selections.

Performance optimizations over the naive single-process version:
- Caches raw policy output per rollout step: the GPU forward pass is called
  ONCE per (state, step), then all 41 groups derive their actions from the
  cached raw output on CPU. This reduces GPU calls from
  41 groups × 10 steps = 410 → just 10 per sample (initial step shared,
  subsequent steps diverge but are cached per unique state).
- Pre-computes group action masks so group-specific post-processing is
  vectorized numpy, not per-group python loops.

The oracle evaluates each candidate group by:
1. Simulating forward N steps using the appropriate policy
2. Accumulating rewards over the horizon
3. Averaging over multiple rollouts
4. Early termination if collision occurs

The collected data format is compatible with train_switcher.py.

Usage:
    python -m robot_nav.scripts.collect_oracle_data_batch

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
import time
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

from robot_nav.models.MARL.groups.group_generator import (
    generate_all_groups,
    filter_groups_by_size,
)
from robot_nav.models.MARL.groups.action_coupling import (
    actions_for_group_from_raw as _shared_actions_for_group_from_raw,
    actions_for_group as _shared_actions_for_group,
)

# =============================================================================
# Configuration Dictionary - Edit these values directly
# =============================================================================
CONFIG = {
    # Output configuration
    "output_path": "robot_nav/models/MARL/switcher/data/oracle_data_14robots_decouple_couple_group_len1200_success.pt",

    # Data collection settings
    "n_samples": 15000,              # Number of samples to collect
    "n_robots": 14,                  # Number of robots
    "n_obstacles": 7,               # Number of obstacles
    "embed_dim": 512,               # Per-robot embedding dimension from GAT backbone output (2*embedding_dim=2*256)
    "seed": 42,                     # Random seed for reproducibility
    
    # Oracle evaluation settings
    "oracle_horizon": 10,           # Number of steps to simulate forward for each group
    "n_rollouts_per_group": 1,      # Number of rollouts to average for each group score
    
    # Group generation settings
    "include_size_1": True,         # Include individual robots as candidates
    "include_size_2": True,         # Include pairs
    "include_size_3": True,         # Include triplets
    "include_size_4": True,        # Include size-4 groups (rotation-coupled if enabled)
    "include_size_7": True,        # Include size-7 groups (rotation-coupled if enabled)
    
    # Rotation coupling for large groups (size > 3):
    # When True, groups with size > 3 use coupled rotation in addition to
    # coupled linear velocity. All robots in the group rotate at the same
    # angular velocity (average of individual angular velocities) and move
    # at the same linear velocity (minimum of individual linear velocities).
    # Groups with size <= 3 always keep individual angular velocities.
    "use_rotation_coupling": True,
    
    # Model configuration
    "state_dim": 11,
    "obstacle_state_dim": 4,
    
    # Pretrained model paths (decentralized TD3Obstacle policy)
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-gpu_epoch800",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Feb.10_obstacle_14robot_transfer_gpu",

    # Simulation settings
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "disable_plotting": True,
    "obstacle_proximity_threshold": 1.5,
    "max_steps_per_episode": 1200,   # Reset episode after this many steps
    
    # Goal reach threshold (matches world yaml goal_threshold)
    "goal_reach_threshold": 0.3,
    
    # Scoring weights
    "k_reach": 50.0,               # New-reach bonus weight
    "k_progress": 3.0,             # Progress reward weight (translational)
    "k_rotation_progress": 2.0,    # Rotation progress reward weight
    "k_sync": 3.0,                 # Synchronization reward weight (only active when ≥1 robot reached)
    "k_urgency": 15.0,             # Urgency bonus weight for moving stuck robots (only if use_urgency=True)
    
    # Stuckness detection thresholds
    "min_displacement_threshold": 0.2,      # Minimum translational displacement to avoid stuckness penalty
    "min_rotation_threshold": 0.1,          # Minimum average rotation (rad) to avoid stuckness penalty
    
    # Urgency tracking (for stuck robot detection, only if use_urgency=True)
    "use_urgency": False,             # Enable/disable urgency tracking and bonus for stuck robots
    "urgency_lookback_window": 20,          # Number of recent oracle selections to track per robot
    "urgency_stuck_threshold": 0.3,         # If robot moved < this distance over lookback window, it's stuck
    
    # Phase 2 group selection strategy:
    #   "random"  — uniformly random group (original behavior)
    #   "softmax" — sample from all groups weighted by softmax of oracle scores
    "phase2_selection": "softmax",
    "phase2_temperature": 1.0,     # Softmax temperature: ~0 = always pick max, higher = more uniform
}


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
    - For size-2/3 groups: min linear velocities of robots in the group
      to get coupled linear velocity, keep individual angular velocities
    - For size-4/7 groups (rotation-coupled, if use_rotation_coupling=True):
      min linear velocity AND average angular velocity — all robots in the
      group move and rotate at the same speed
    - For size-4/7 groups (if use_rotation_coupling=False):
      same as size-2/3 (coupled linear only, individual angular)
    
    This follows the same pattern as ShortHorizonOracle in coupled_action_oracle_eval.py.
    """
    
    def __init__(
        self,
        sim,                          # MARL_SIM_OBSTACLE instance
        policy,                       # TD3Obstacle policy (decentralized)
        groups: List[List[int]],
        horizon: int = 10,            # Number of steps to simulate forward
        n_rollouts_per_group: int = 3,
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
    
    def get_raw_actions(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Run the policy ONCE and return the raw per-robot actions.
        
        This is the expensive GPU forward pass. Call it once per unique
        observation, then derive group-specific actions on CPU with
        actions_for_group_from_raw().
        
        Args:
            robot_obs: Robot observations, shape (num_robots, state_dim).
            obstacle_obs: Obstacle observations, shape (num_obstacles, obs_dim).
            
        Returns:
            raw_actions: np.ndarray of shape (num_robots, 2) with values in [-1, 1].
        """
        action, _ = self.policy.get_action(
            robot_obs, obstacle_obs, add_noise=False
        )
        return action  # (num_robots, 2)
    
    def get_raw_actions_batched(
        self,
        robot_obs_batch: np.ndarray,
        obstacle_obs_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Run the policy on a BATCH of observations in a single GPU forward pass.
        
        This is the key optimization: instead of calling the GPU N_groups times
        (once per group per step), we batch all groups' observations into one
        call, reducing N_groups GPU calls → 1 GPU call.
        
        Args:
            robot_obs_batch: Batched robot observations, shape (B, num_robots, state_dim).
            obstacle_obs_batch: Batched obstacle observations, shape (B, num_obstacles, obs_dim).
            
        Returns:
            raw_actions_batch: np.ndarray of shape (B, num_robots, 2) with values in [-1, 1].
        """
        B = robot_obs_batch.shape[0]
        
        robot_tensor = torch.tensor(robot_obs_batch, dtype=torch.float32, device=self.device)
        obstacle_tensor = torch.tensor(obstacle_obs_batch, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # actor forward: robot_obs (B, N, state_dim), obstacle_obs (B, N_obs, obs_dim)
            # returns action of shape (B*N, action_dim)
            action_flat, *_ = self.policy.actor(robot_tensor, obstacle_tensor)
        
        # Reshape from (B*N, action_dim) to (B, N, action_dim)
        N = self.num_robots
        actions = action_flat.cpu().numpy().reshape(B, N, -1)
        return actions  # (B, num_robots, 2)
    
    def actions_for_group_from_raw(
        self,
        raw_actions: np.ndarray,
        group: List[int],
    ) -> List[List[float]]:
        """Derive group-specific actions from pre-computed raw policy output.
        
        Delegates to ``robot_nav.models.MARL.groups.action_coupling``.
        """
        return _shared_actions_for_group_from_raw(
            raw_actions=raw_actions,
            group=group,
            num_robots=self.num_robots,
            use_rotation_coupling=CONFIG.get("use_rotation_coupling", True),
            rotation_coupling_threshold=3,
        )
    
    def get_action_for_group(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        group: List[int],
    ) -> List[List[float]]:
        """Get action for a specific group using the decentralized policy.
        
        Delegates to ``robot_nav.models.MARL.groups.action_coupling``.
        """
        return _shared_actions_for_group(
            policy=self.policy,
            robot_obs=robot_obs,
            obstacle_obs=obstacle_obs,
            group=group,
            num_robots=self.num_robots,
            use_rotation_coupling=CONFIG.get("use_rotation_coupling", True),
            rotation_coupling_threshold=3,
        )
    
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
        min_rotation_threshold: float = 0.1,
        had_new_reach: bool = False,
        reached: Optional[List[bool]] = None,
    ) -> float:
        """
        Compute penalty for groups that result in very low movement (stuckness).
        
        This discourages the switcher from selecting groups that don't make progress.
        Skips already-reached robots (they may orbit near goal, that's OK).
        
        A group is NOT stuck if either:
        - Average translational displacement >= min_displacement_threshold, OR
        - Average rotation >= min_rotation_threshold (rotating towards goal is progress)
        
        Args:
            group: Robot indices in the group
            initial_poses: Per-robot poses at start [[x, y, theta], ...]
            final_poses: Per-robot poses at end [[x, y, theta], ...]
            min_displacement_threshold: Minimum expected displacement over horizon
            min_rotation_threshold: Minimum expected rotation (rad) over horizon
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
        total_rotation = 0.0
        for i in unreached_in_group:
            xi_init, yi_init, theta_init = initial_poses[i]
            xi_final, yi_final, theta_final = final_poses[i]
            
            # Translational displacement
            displacement = np.sqrt((xi_final - xi_init)**2 + (yi_final - yi_init)**2)
            total_displacement += displacement
            
            # Rotational displacement (absolute angle change)
            angle_diff = np.abs(theta_final - theta_init)
            # Normalize to [0, pi] (we care about magnitude of rotation, not direction)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            total_rotation += angle_diff
        
        avg_displacement = total_displacement / len(unreached_in_group)
        avg_rotation = total_rotation / len(unreached_in_group)
        
        # Group is NOT stuck if EITHER translational OR rotational movement is sufficient
        is_stuck = (avg_displacement < min_displacement_threshold) and (avg_rotation < min_rotation_threshold)
        
        # Penalize if stuck (both displacement AND rotation below threshold)
        # Penalty is based ONLY on displacement deficit, rotation is only used to determine if stuck
        if is_stuck:
            displacement_deficit = max(0, min_displacement_threshold - avg_displacement)
            return -k_stuck * displacement_deficit
        
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
        goal_positions: List[List[float]],
        robot_proximity_threshold: float = 1.5,
        obstacle_proximity_threshold: float = 1.5,
        min_displacement_threshold: float = 0.2,
        urgency_flags: Optional[List[bool]] = None,
    ) -> float:
        """
        Compute trajectory-based score for an oracle rollout.
        
        Revised scoring for synchronized goal arrival:
        
        1. Collision penalty: -50 if any collision occurred
        2. New-reach bonus: k_reach / n_remaining per newly reached robot
           (more valuable when fewer robots remain unreached)
        3. Progress reward: Laggard-weighted — robots farther from goal get
           more reward per meter of progress. Already-reached robots excluded.
        4. Rotation progress reward: Reward for aligning heading with goal direction.
        5. Urgency bonus: Extra reward for SINGLE-ROBOT groups (size 1) that move
           stuck robots. Only applies when the robot has been stuck for a long time
           (haven't made progress in recent oracle selections). Multi-robot groups
           don't get this bonus as they dilute the targeted help.
        6. Synchronization reward: Reduces variance of dist_to_goal across
           ALL robots — only active when ≥1 robot has reached goal.
        7. Evasion reward: Unchanged from original.
        8. Stuckness penalty: Unchanged, but skips already-reached robots.
        
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
            urgency_flags: Per-robot urgency flags (True = robot has been stuck for a while)
            
        Returns:
            score: Trajectory score (higher = better)
        """
        N = self.num_robots
        k_reach = CONFIG.get("k_reach", 50.0)
        k_progress = CONFIG.get("k_progress", 3.0)
        k_sync = CONFIG.get("k_sync", 8.0)
        k_urgency = CONFIG.get("k_urgency", 15.0)
        
        # 1. Collision penalty
        if had_collision:
            return -50.0

        # Extra score for coupled group (size > 3)
        # Only give bonus if NO robots in the group have already reached
        # (coupling doesn't make sense if some robots are already at goal)
        any_in_group_reached = any(reached_before_rollout[i] for i in group)
        coupled_group_score = 5.0 if (len(group) > 3 and not any_in_group_reached) else 0.0
        score = coupled_group_score
        
        # 2. New-reach bonus: k_reach / n_remaining per newly reached robot
        # The fewer robots remaining, the more valuable each new reach is.
        if n_new_reached > 0:
            n_remaining_before = N - n_already_reached_before
            if n_remaining_before > 0:
                # Each new reach gets bonus proportional to 1/n_remaining
                # As robots progressively reach, each subsequent reach is worth more
                for r in range(n_new_reached):
                    remaining_at_time = n_remaining_before - r
                    if remaining_at_time > 0:
                        score += k_reach / remaining_at_time
        
        # 3. Laggard-weighted progress reward (translational)
        # Only for unreached robots. Weight = dist[i] / mean_dist (farther = more reward)
        unreached_indices = [
            i for i in range(N) if not reached_before_rollout[i]
        ]
        if len(unreached_indices) > 0:
            unreached_dists = [initial_distances[i] for i in unreached_indices]
            mean_dist = np.mean(unreached_dists) if len(unreached_dists) > 0 else 1.0
            mean_dist = max(mean_dist, 0.1)  # avoid division by zero
            
            progress_reward = 0.0
            for i in group:
                if reached_before_rollout[i]:
                    continue  # skip already-reached robots
                progress = initial_distances[i] - final_distances[i]
                laggard_weight = initial_distances[i] / mean_dist
                progress_reward += k_progress * progress * laggard_weight
            score += progress_reward
        
        # 3b. Rotation progress reward: rotating towards goal is also progress
        k_rotation_progress = CONFIG.get("k_rotation_progress", 2.0)
        rotation_progress_reward = 0.0
        for i in group:
            if reached_before_rollout[i]:
                continue
            
            xi_init, yi_init, theta_init = initial_poses[i]
            xi_final, yi_final, theta_final = final_poses[i]
            
            # Get goal positions for robot i
            goal_x, goal_y = goal_positions[i]
            
            # Compute heading error at initial and final states
            # Heading error = angle between robot heading and direction to goal
            dx_init = goal_x - xi_init
            dy_init = goal_y - yi_init
            angle_to_goal_init = np.arctan2(dy_init, dx_init)
            heading_error_init = np.abs(theta_init - angle_to_goal_init)
            # Normalize to [0, pi]
            heading_error_init = min(heading_error_init, 2*np.pi - heading_error_init)
            
            dx_final = goal_x - xi_final
            dy_final = goal_y - yi_final
            angle_to_goal_final = np.arctan2(dy_final, dx_final)
            heading_error_final = np.abs(theta_final - angle_to_goal_final)
            # Normalize to [0, pi]
            heading_error_final = min(heading_error_final, 2*np.pi - heading_error_final)
            
            # Reward for reducing heading error (aligning with goal direction)
            heading_improvement = heading_error_init - heading_error_final
            rotation_progress_reward += k_rotation_progress * heading_improvement
        
        score += rotation_progress_reward
        
        # 4. Urgency bonus: reward SINGLE-ROBOT groups that move stuck robots
        # Only size-1 groups get this bonus - they're specifically helping that robot.
        # Disabled when use_urgency=False in CONFIG.
        urgency_bonus = 0.0
        if CONFIG.get("use_urgency", True) and urgency_flags is not None and len(group) == 1:
            robot_idx = group[0]
            if not reached_before_rollout[robot_idx] and urgency_flags[robot_idx]:
                # This single-robot group is moving a stuck robot
                # Reward based on displacement (movement) rather than just goal progress
                # A stuck robot needs to get unstuck first, even if it means moving
                # away from goal temporarily
                xi_init, yi_init, _ = initial_poses[robot_idx]
                xi_final, yi_final, _ = final_poses[robot_idx]
                displacement = np.sqrt((xi_final - xi_init)**2 + (yi_final - yi_init)**2)
                
                # Also consider goal progress (if moving toward goal, that's even better)
                progress = initial_distances[robot_idx] - final_distances[robot_idx]
                
                # Give bonus for displacement (getting unstuck), with extra bonus if
                # also making goal progress
                if displacement > 0.01:  # Small threshold to filter noise
                    urgency_bonus = k_urgency * displacement
                    # Extra bonus if progress is positive (moving toward goal)
                    if progress > 0:
                        urgency_bonus += k_urgency * progress * 0.5
        score += urgency_bonus
        
        # 5. Synchronization reward
        # Only activate when at least one robot has reached goal.
        # At the beginning when all robots just start, synchronization doesn't make sense.
        # Once some robots reach, we want to encourage others to catch up (reduce variance).
        sync_reward = 0.0
        any_robot_reached = any(reached_before_rollout)
        if N >= 2 and any_robot_reached:
            initial_all_dists = np.array(initial_distances)
            final_all_dists = np.array(final_distances)
            var_before = np.var(initial_all_dists)
            var_after = np.var(final_all_dists)
            sync_reward = k_sync * (var_before - var_after)
        score += sync_reward
        
        # 6. Evasion reward: reward for rotating/moving away from nearby entities
        evasion_reward, _ = self.compute_evasion_reward(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            initial_obstacle_states=initial_obstacle_states,
            final_obstacle_states=final_obstacle_states,
            robot_proximity_threshold=robot_proximity_threshold,
            obstacle_proximity_threshold=obstacle_proximity_threshold,
            return_details=False,
        )
        score += evasion_reward
        
        # 6. Stuckness penalty: penalize groups with very low displacement
        min_rotation_threshold = CONFIG.get("min_rotation_threshold", 0.1)
        stuckness_penalty = self.compute_stuckness_penalty(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            min_displacement_threshold=min_displacement_threshold,
            min_rotation_threshold=min_rotation_threshold,
            had_new_reach=(n_new_reached > 0),
            reached=reached_before_rollout,
        )
        score += stuckness_penalty
        
        return score
    
    def _restore_group_state_to_sim(
        self,
        gs: Dict,
    ):
        """
        Restore a group's diverged state into the simulation.
        
        This sets robot positions, goals, and prev_distances so that
        sim.step() will produce correct physics from this state.
        
        Args:
            gs: Group state dict with keys 'poses', 'goal_positions', 'distance'.
        """
        for i, robot in enumerate(self.sim.env.robot_list):
            # Reconstruct state as (3, 1) column vector matching robot.state format
            pose = gs["poses"][i]
            state_arr = np.array([[pose[0]], [pose[1]], [pose[2]]], dtype=np.float64)
            robot.set_state(state=state_arr, init=True)
            
            # Reconstruct goal as (3, 1) column vector matching robot.goal format
            gp = gs["goal_positions"][i]
            goal_arr = np.array([[gp[0]], [gp[1]], [0.0]], dtype=np.float64)
            robot.set_goal(goal_arr, init=True)
        
        self.sim.prev_distances = gs["distance"].copy()
    
    def _evaluate_all_groups_batched(
        self,
        poses: List[List[float]],
        distance: List[float],
        cos_: List[float],
        sin_: List[float],
        collision: List[bool],
        action: List[List[float]],
        goal_positions: List[List[float]],
        obstacle_states: np.ndarray,
        snapshot: SimulationSnapshot,
        reached: Optional[List[bool]] = None,
        urgency_flags: Optional[List[bool]] = None,
    ) -> List[float]:
        """
        Evaluate ALL groups using batched GPU inference.
        
        Instead of evaluating groups one-by-one (each needing its own GPU calls),
        this method runs all groups' rollouts together. At each horizon step:
        
        1. Collect observations from all active (non-terminated) groups
        2. Batch them into a single GPU forward pass → get raw actions for all groups
        3. For each group, derive group-specific actions and step the sim
        
        Tracks sticky reached[] flags per group rollout to detect new reaches.
        
        Args:
            poses, distance, cos_, sin_, collision, action, goal_positions, obstacle_states:
                Current environment state (shared starting point for all groups).
            snapshot: Simulation snapshot to restore after each group's rollout step.
            reached: Episode-level sticky reached flags per robot (None = all False).
            urgency_flags: Per-robot urgency flags (stuck robots).
            
        Returns:
            scores: List of float scores, one per group, in self.groups order.
        """
        N_groups = len(self.groups)
        N_robots = self.num_robots
        goal_threshold = CONFIG.get("goal_reach_threshold", 0.3)
        
        # Episode-level reached state (before any rollout)
        if reached is None:
            reached = [False] * N_robots
        reached_before = list(reached)  # snapshot for scoring
        n_already_reached = sum(reached_before)
        
        # Per-group rollout state (maintained independently for each group)
        # Initialize all groups from the same starting state
        group_states = []
        for g_idx in range(N_groups):
            group_states.append({
                "poses": [list(p) for p in poses],
                "distance": list(distance),
                "cos": list(cos_),
                "sin": list(sin_),
                "collision": list(collision),
                "action": [list(a) for a in action],
                "goal_positions": [list(g) for g in goal_positions],
                "obstacle_states": obstacle_states.copy(),
                "had_collision": False,
                "terminated": False,
                "initial_poses": [list(p) for p in poses],
                "initial_distances": list(distance),
                "initial_obstacle_states": obstacle_states.copy(),
                # Per-group copy of reached flags (sticky within rollout)
                "reached": list(reached_before),
                "n_new_reached": 0,
            })
        
        for step in range(self.horizon):
            # Collect indices of groups that are still active (not terminated)
            active_indices = [
                g_idx for g_idx in range(N_groups) 
                if not group_states[g_idx]["terminated"]
            ]
            
            if len(active_indices) == 0:
                break  # All groups terminated early
            
            # === BATCHED OBSERVATION PREPARATION ===
            # Prepare robot observations for all active groups
            robot_obs_list = []
            obstacle_obs_list = []
            for g_idx in active_indices:
                gs = group_states[g_idx]
                robot_state, _ = self.policy.prepare_state(
                    gs["poses"], gs["distance"], gs["cos"], gs["sin"],
                    gs["collision"], gs["action"], gs["goal_positions"],
                )
                robot_obs_list.append(np.array(robot_state))
                obstacle_obs_list.append(gs["obstacle_states"])
            
            # Stack into batched arrays: (B, N_robots, state_dim) and (B, N_obs, obs_dim)
            robot_obs_batch = np.stack(robot_obs_list, axis=0)
            obstacle_obs_batch = np.stack(obstacle_obs_list, axis=0)
            
            # === SINGLE BATCHED GPU FORWARD PASS ===
            raw_actions_batch = self.get_raw_actions_batched(
                robot_obs_batch, obstacle_obs_batch
            )  # (B, N_robots, 2)
            
            # === PER-GROUP SIM STEPPING (sequential, sim is shared) ===
            for batch_idx, g_idx in enumerate(active_indices):
                gs = group_states[g_idx]
                group = self.groups[g_idx]
                raw_actions = raw_actions_batch[batch_idx]  # (N_robots, 2)
                
                # Derive group-specific actions from raw actions (CPU only)
                a_in = self.actions_for_group_from_raw(raw_actions, group)
                
                # Restore sim to this group's current diverged state
                self._restore_group_state_to_sim(gs)
                
                # Step simulation
                (
                    new_poses, new_distance, new_cos, new_sin,
                    new_collision, new_goal, new_action, reward,
                    _, new_goal_positions, new_obstacle_states
                ) = self.sim.step(a_in, None, None)
                
                # Update group state
                gs["poses"] = new_poses
                gs["distance"] = new_distance
                gs["cos"] = new_cos
                gs["sin"] = new_sin
                gs["collision"] = list(new_collision)
                gs["action"] = new_action
                gs["goal_positions"] = new_goal_positions
                gs["obstacle_states"] = new_obstacle_states
                
                # Check for NEW goal reaches (sticky flag)
                # A robot is "reached" if distance < threshold and not already flagged
                for i in range(N_robots):
                    if not gs["reached"][i] and new_distance[i] < goal_threshold:
                        gs["reached"][i] = True
                        gs["n_new_reached"] += 1
                
                # Check for collision - terminate this group
                if any(new_collision[i] for i in group):
                    gs["had_collision"] = True
                    gs["terminated"] = True
                
                # Check for out of bounds
                if outside_of_bounds(new_poses, self.sim):
                    gs["had_collision"] = True
                    gs["terminated"] = True
        
        # === COMPUTE SCORES FOR ALL GROUPS ===
        scores = []
        for g_idx in range(N_groups):
            gs = group_states[g_idx]
            group = self.groups[g_idx]
            
            final_poses = gs["poses"]
            final_distances = gs["distance"]
            final_obstacle_states = gs["obstacle_states"]
            
            trajectory_score = self.compute_trajectory_score(
                group=group,
                initial_poses=gs["initial_poses"],
                final_poses=final_poses,
                initial_distances=gs["initial_distances"],
                final_distances=final_distances,
                initial_obstacle_states=gs["initial_obstacle_states"],
                final_obstacle_states=final_obstacle_states,
                had_collision=gs["had_collision"],
                n_new_reached=gs["n_new_reached"],
                n_already_reached_before=n_already_reached,
                reached_before_rollout=reached_before,
                goal_positions=goal_positions,
                robot_proximity_threshold=1.5,
                obstacle_proximity_threshold=self.sim.obstacle_proximity_threshold,
                min_displacement_threshold=0.2,
                urgency_flags=urgency_flags,
            )
            scores.append(trajectory_score)
        
        # Restore sim to original snapshot
        snapshot.restore_to_sim(self.sim)
        
        return scores
    
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
        precomputed_raw_actions: Optional[np.ndarray] = None,
        reached: Optional[List[bool]] = None,
        urgency_flags: Optional[List[bool]] = None,
    ) -> float:
        """
        Evaluate a single group (fallback, used when n_rollouts_per_group > 1).
        
        For the common case of n_rollouts_per_group == 1, prefer using
        _evaluate_all_groups_batched() which is much faster.
        
        Args:
            group: List of robot indices in the group
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after each rollout.
            precomputed_raw_actions: Optional pre-computed raw actions for step 0.
            reached: Episode-level sticky reached flags per robot.
            urgency_flags: Per-robot urgency flags (stuck robots).
            
        Returns:
            score: Average cumulative reward across rollouts (higher = better)
        """
        N = self.num_robots
        goal_threshold = CONFIG.get("goal_reach_threshold", 0.3)
        
        if reached is None:
            reached = [False] * N
        reached_before = list(reached)
        n_already_reached = sum(reached_before)
        
        total_reward = 0.0
        
        for rollout_idx in range(self.n_rollouts_per_group):
            had_collision = False
            n_new_reached = 0
            rollout_reached = list(reached_before)
            
            initial_poses = [p.copy() for p in poses]
            initial_distances = distance.copy()
            initial_obstacle_states = obstacle_states.copy()
            
            curr_poses = [p.copy() for p in poses]
            curr_distance = distance.copy()
            curr_cos = cos.copy()
            curr_sin = sin.copy()
            curr_collision = list(collision)
            curr_action = [a.copy() for a in action]
            curr_goal_positions = [g.copy() for g in goal_positions]
            curr_obstacle_states = obstacle_states.copy()
            
            for step in range(self.horizon):
                if step == 0 and precomputed_raw_actions is not None:
                    raw_actions = precomputed_raw_actions
                else:
                    robot_state, _ = self.policy.prepare_state(
                        curr_poses, curr_distance, curr_cos, curr_sin,
                        curr_collision, curr_action, curr_goal_positions
                    )
                    raw_actions = self.get_raw_actions(
                        np.array(robot_state), curr_obstacle_states
                    )
                
                a_in = self.actions_for_group_from_raw(raw_actions, group)
                
                (
                    curr_poses, curr_distance, curr_cos, curr_sin,
                    curr_collision, curr_goal, curr_action, reward,
                    _, curr_goal_positions, curr_obstacle_states
                ) = self.sim.step(a_in, None, None)
                
                # Check for new reaches (sticky)
                for i in range(N):
                    if not rollout_reached[i] and curr_distance[i] < goal_threshold:
                        rollout_reached[i] = True
                        n_new_reached += 1
                
                if any(curr_collision[i] for i in group):
                    had_collision = True
                    break
                
                if outside_of_bounds(curr_poses, self.sim):
                    had_collision = True
                    break
            
            trajectory_score = self.compute_trajectory_score(
                group=group,
                initial_poses=initial_poses,
                final_poses=[p.copy() for p in curr_poses],
                initial_distances=initial_distances,
                final_distances=curr_distance.copy(),
                initial_obstacle_states=initial_obstacle_states,
                final_obstacle_states=curr_obstacle_states.copy(),
                had_collision=had_collision,
                n_new_reached=n_new_reached,
                n_already_reached_before=n_already_reached,
                reached_before_rollout=reached_before,
                goal_positions=goal_positions,
                robot_proximity_threshold=1.5,
                obstacle_proximity_threshold=self.sim.obstacle_proximity_threshold,
                min_displacement_threshold=0.2,
                urgency_flags=urgency_flags,
            )
            
            snapshot.restore_to_sim(self.sim)
            total_reward += trajectory_score
        
        return total_reward / self.n_rollouts_per_group
    
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
            h: Per-robot embeddings, Tensor[N, embed_dim]
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
                H,  # Per-robot embeddings: (B*N, embed_dim)
                hard_logits_rr,
                hard_logits_ro,
                dist_rr,
                dist_ro,
                mean_entropy,
                hard_weights_rr,  # (B, N, N)
                hard_weights_ro,  # (B, N, N_obs)
                combined_weights,
            ) = self.policy.actor.attention(robot_tensor, obstacle_tensor)
        
        # Reshape H from (B*N, embed_dim) to (B, N, embed_dim), then remove batch dim
        batch_size = robot_tensor.shape[0]
        n_robots = robot_tensor.shape[1]
        embed_dim = H.shape[-1]  # embed_dim (= 2 * GAT embedding_dim = 512)
        
        h = H.view(batch_size, n_robots, embed_dim).squeeze(0)  # (N, embed_dim)
        attn_rr = hard_weights_rr.squeeze(0)  # (N, N)
        attn_ro = hard_weights_ro.squeeze(0)  # (N, N_obs)
        
        return h, attn_rr, attn_ro
    
    def _get_extra_features(
        self,
        poses: List[List[float]],
        distance: List[float],
        goal_positions: List[List[float]],
        reached: Optional[List[bool]] = None,
        urgency_flags: Optional[List[bool]] = None,
        step_in_episode: int = 0,
        max_steps: int = 400,
    ) -> Dict[str, torch.Tensor]:
        """
        Get extra features from environment state for GroupFeatureBuilder.
        
        Returns two kinds of tensors:
        
        Per-robot tensors (shape ``(N,)``) — GroupFeatureBuilder aggregates
        these over the robots *in each group*:
          - ``dist_to_goal``   → mean → mean_dist_goal_g   (feat 6)
          -                    → min  → min_dist_goal_g    (feat 7)
          - ``clearance``      → min  → min_clearance_g    (feat 8)
          - ``reached``        → mean → frac_reached_g     (feat 9)
          - ``heading_error``  → mean → mean_heading_err_g (feat 10)
          - ``urgency``        → mean → frac_urgent_g      (feat 11)
        
        Global scalars (shape ``(1,)``) — GroupFeatureBuilder broadcasts
        the same value to every group:
          - ``var_dist_goal_global``  — distance variance  (feat 12)
          - ``frac_reached_global``   — completion fraction (feat 13)
          - ``steps_elapsed_frac``    — time pressure       (feat 14)
        
        Args:
            poses: Per-robot poses [[x, y, theta], ...]
            distance: Per-robot distances to goal
            goal_positions: Per-robot goals [[gx, gy], ...]
            reached: Per-robot sticky reached flags
            urgency_flags: Per-robot urgency flags (stuck robots)
            step_in_episode: Current step in the episode
            max_steps: Maximum steps per episode
            
        Returns:
            extra: Dict with per-robot and global feature tensors.
        """
        N = self.num_robots
        
        # ── Per-robot tensors (N,) ──
        
        # Distance to goal
        dist_to_goal = torch.tensor(distance, dtype=torch.float32)
        
        # Minimum clearance to obstacles
        clearances = []
        for i in range(N):
            min_clearance = self.sim.get_min_obstacle_clearance(i)
            clearances.append(min_clearance)
        clearance = torch.tensor(clearances, dtype=torch.float32)
        
        # Reached flags (0.0 / 1.0)
        if reached is None:
            reached = [False] * N
        reached_float = torch.tensor(
            [1.0 if r else 0.0 for r in reached], dtype=torch.float32
        )
        
        # Heading error: |heading − goal_direction| in radians
        heading_errors = []
        for i in range(N):
            x, y, theta = poses[i][0], poses[i][1], poses[i][2]
            gx, gy = goal_positions[i][0], goal_positions[i][1]
            goal_dir = np.arctan2(gy - y, gx - x)
            diff = goal_dir - theta
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            heading_errors.append(abs(diff))
        heading_error = torch.tensor(heading_errors, dtype=torch.float32)
        
        # Urgency flags (per-robot binary indicator for stuck robots)
        # Always zeros when use_urgency=False in CONFIG
        if urgency_flags is None or not CONFIG.get("use_urgency", True):
            urgency_flags = [False] * N
        urgency_float = torch.tensor(
            [1.0 if u else 0.0 for u in urgency_flags], dtype=torch.float32
        )
        
        # ── Global scalars (1,) ──
        
        # Variance of dist_to_goal across all robots (sync signal)
        unreached_dists = [distance[i] for i in range(N) if not reached[i]]
        if len(unreached_dists) >= 2:
            var_dist_val = float(np.var(unreached_dists))
        else:
            var_dist_val = 0.0
        var_dist_goal_global = torch.tensor([var_dist_val], dtype=torch.float32)
        
        # Fraction of all robots that have reached
        frac_reached_global = torch.tensor(
            [sum(reached) / N], dtype=torch.float32
        )
        
        # Steps elapsed fraction
        steps_frac = step_in_episode / max(max_steps, 1)
        steps_elapsed_frac = torch.tensor([steps_frac], dtype=torch.float32)
        
        return {
            # Per-robot (N,)
            "dist_to_goal": dist_to_goal,
            "clearance": clearance,
            "reached": reached_float,
            "heading_error": heading_error,
            "urgency": urgency_float,
            # Global (1,)
            "var_dist_goal_global": var_dist_goal_global,
            "frac_reached_global": frac_reached_global,
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
        urgency_flags: Optional[List[bool]] = None,
        step_in_episode: int = 0,
        max_steps: int = 400,
    ) -> Dict:
        """
        Collect one sample of oracle data at the current simulation state.
        
        Optimized flow (batched GPU inference):
        1. Prepare observations ONCE
        2. Get embeddings and attention (1 GPU call)
        3. Evaluate ALL groups using batched rollouts with reached tracking
        4. Return sample dict with new extra features
        
        Args:
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state from sim.step() or sim.reset()
            scenario_id: Optional identifier for this sample
            reached: Episode-level sticky reached flags per robot
            urgency_flags: Per-robot urgency flags (stuck robots)
            step_in_episode: Current step in the episode
            max_steps: Maximum steps per episode
            
        Returns:
            Sample dictionary compatible with train_switcher.py
        """
        # Take snapshot before any rollouts
        snapshot = SimulationSnapshot.from_sim(self.sim)
        
        # Prepare robot observations ONCE (used for embeddings)
        robot_state, _ = self.policy.prepare_state(
            poses, distance, cos, sin, collision, action, goal_positions
        )
        robot_obs = np.array(robot_state)
        
        # === GPU CALL: Get embeddings and attention (1 forward pass) ===
        h, attn_rr, attn_ro = self._get_embeddings_and_attention(robot_obs, obstacle_states)
        
        # Get extra features (CPU, fast) — now includes reached, sync features, urgency
        extra = self._get_extra_features(
            poses, distance, goal_positions,
            reached=reached,
            urgency_flags=urgency_flags,
            step_in_episode=step_in_episode,
            max_steps=max_steps,
        )
        
        # === BATCHED GROUP EVALUATION ===
        if self.n_rollouts_per_group == 1:
            # Fast path: batched evaluation for all groups simultaneously
            group_scores_list = self._evaluate_all_groups_batched(
                poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states, snapshot,
                reached=reached,
                urgency_flags=urgency_flags,
            )
        else:
            # Fallback: sequential per-group evaluation with multiple rollouts
            precomputed_raw_actions = self.get_raw_actions(robot_obs, obstacle_states)
            group_scores_list = []
            for group in self.groups:
                score = self._evaluate_group(
                    group, poses, distance, cos, sin, collision, action,
                    goal_positions, obstacle_states, snapshot,
                    precomputed_raw_actions=precomputed_raw_actions,
                    reached=reached,
                    urgency_flags=urgency_flags,
                )
                group_scores_list.append(score)
        
        group_scores = torch.tensor(group_scores_list, dtype=torch.float32)
        
        return {
            "h": h.cpu(),
            "groups": self.groups,
            "group_scores": group_scores,
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
        samples = []
        
        pbar = tqdm(range(n_samples), desc="Collecting oracle data") if verbose else range(n_samples)
        
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
        
        # Urgency tracking: maintain a sliding window of recent distances per robot
        use_urgency = CONFIG.get("use_urgency", True)
        urgency_lookback = CONFIG.get("urgency_lookback_window", 20)
        urgency_stuck_threshold = CONFIG.get("urgency_stuck_threshold", 0.3)
        robot_distance_history = [[] for _ in range(N)]  # List of lists: per-robot distance history
        urgency_flags = [False] * N  # Current urgency flags
        
        # Statistics
        t_start = time.time()
        gpu_calls = 0
        episode_count = 0
        all_reached_count = 0  # Episodes where all robots reached
        collision_reset_count = 0  # Episodes reset due to collision
        oob_reset_count = 0  # Episodes reset due to out-of-bounds
        timeout_reset_count = 0  # Episodes reset due to max steps
        total_oracle_groups_evaluated = 0  # Total group rollouts across all samples
        total_oracle_collisions = 0  # Group rollouts that ended in collision (score == -50)
        discarded_samples = 0  # Samples discarded from failed trajectories
        
        # Episode buffer: collect samples during episode, only commit if successful
        episode_buffer = []
        
        # Phase 2 group selection tracking
        phase2_group_counts = {i: 0 for i in range(len(self.groups))}
        
        for i in pbar:
            # Update urgency tracking: check if robots have been stuck
            # Skip entirely when urgency is disabled
            if not use_urgency:
                urgency_flags = [False] * N
            for robot_idx in range(N) if use_urgency else []:
                if not reached[robot_idx]:
                    robot_distance_history[robot_idx].append(distance[robot_idx])
                    # Keep only recent lookback_window samples
                    if len(robot_distance_history[robot_idx]) > urgency_lookback:
                        robot_distance_history[robot_idx].pop(0)
                    
                    # Compute urgency: if robot has full history and made little progress
                    if len(robot_distance_history[robot_idx]) >= urgency_lookback:
                        oldest_dist = robot_distance_history[robot_idx][0]
                        current_dist = robot_distance_history[robot_idx][-1]
                        progress_over_window = oldest_dist - current_dist
                        urgency_flags[robot_idx] = (progress_over_window < urgency_stuck_threshold)
                    else:
                        urgency_flags[robot_idx] = False
                else:
                    # Robot has reached goal - clear urgency tracking
                    urgency_flags[robot_idx] = False
                    # Clear distance history to prevent stale data if robot un-reaches somehow
                    robot_distance_history[robot_idx] = []
            
            # Collect sample at current state (always collect, decide later whether to keep)
            sample = self.collect_sample(
                poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states,
                scenario_id=len(samples) + len(episode_buffer),  # Total samples that would be saved
                reached=reached,
                urgency_flags=urgency_flags,
                step_in_episode=step_in_episode,
                max_steps=max_steps,
            )
            episode_buffer.append(sample)
            gpu_calls += 1 + self.horizon  # embedding + batched steps
            
            # Track oracle rollout statistics
            scores = sample["group_scores"]
            total_oracle_groups_evaluated += len(scores)
            total_oracle_collisions += int((scores == -50.0).sum().item())
            
            # =================================================================
            # PHASE 2: Advance simulation with a selected group
            #          Run H steps (same horizon as oracle) to get a
            #          meaningfully different configuration for next sample.
            #
            #   Selection strategies:
            #     "random"  — uniform random from all groups
            #     "softmax" — sample weighted by softmax of oracle scores
            # =================================================================
            phase2_mode = CONFIG.get("phase2_selection", "random")
            phase2_temperature = CONFIG.get("phase2_temperature", 1.0)
            
            if phase2_mode == "softmax":
                # Softmax-weighted sampling: higher scores → higher probability
                scores = sample["group_scores"]  # Tensor[n_groups]
                logits = scores / phase2_temperature
                probs = torch.softmax(logits, dim=0)
                chosen_idx = torch.multinomial(probs, num_samples=1).item()
                selected_group = self.groups[chosen_idx]
            else:
                # Original: uniform random
                chosen_idx = random.randint(0, len(self.groups) - 1)
                selected_group = self.groups[chosen_idx]
            
            # Track Phase 2 group selection
            phase2_group_counts[chosen_idx] += 1
            
            
            phase2_horizon = self.horizon  # Same as oracle horizon (e.g. 10)
            
            for ph2_step in range(phase2_horizon):
                # Re-query policy at each step with current observations
                robot_state, _ = self.policy.prepare_state(
                    poses, distance, cos, sin, collision, action, goal_positions
                )
                robot_obs = np.array(robot_state)
                
                # Get raw actions once (GPU), then derive group action (CPU)
                raw_actions = self.get_raw_actions(robot_obs, obstacle_states)
                gpu_calls += 1
                scaled_action = self.actions_for_group_from_raw(raw_actions, selected_group)
                
                # Step simulation
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states
                ) = self.sim.step(scaled_action, None, None)
                
                step_in_episode += 1
                
                # Update sticky reached flags within Phase 2
                for r_idx in range(N):
                    if not reached[r_idx] and distance[r_idx] < goal_threshold:
                        reached[r_idx] = True
                
                # Check for early termination conditions
                if any(collision):
                    break
                if outside_of_bounds(poses, self.sim):
                    break
                if all(reached):
                    break
                if step_in_episode >= max_steps:
                    break
            
            # Check for episode reset conditions
            all_robots_reached = all(reached)
            had_collision = any(collision)
            is_oob = outside_of_bounds(poses, self.sim)
            timeout = step_in_episode >= max_steps
            
            should_reset = had_collision or timeout or is_oob or all_robots_reached
            
            if should_reset:
                episode_count += 1
                
                # Determine if episode was successful (only all_robots_reached counts as success)
                episode_successful = all_robots_reached and not had_collision and not is_oob
                
                if episode_successful:
                    # Success! Commit all buffered samples to the dataset
                    all_reached_count += 1
                    samples.extend(episode_buffer)
                    # Update scenario IDs to be sequential
                    for idx, sample in enumerate(episode_buffer):
                        sample["metadata"]["scenario_id"] = len(samples) - len(episode_buffer) + idx
                else:
                    # Episode failed - discard all buffered samples
                    discarded_samples += len(episode_buffer)
                    
                    # Track failure reasons
                    if had_collision:
                        collision_reset_count += 1
                    if is_oob:
                        oob_reset_count += 1
                    if timeout and not all_robots_reached:
                        timeout_reset_count += 1
                
                # Clear episode buffer for next episode
                episode_buffer = []
                
                # Full reset: all robots get new random positions and goals
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states
                ) = self.sim.reset(random_obstacles=True)
                step_in_episode = 0
                reached = [False] * N  # Reset all reached flags
                # Reset urgency tracking
                robot_distance_history = [[] for _ in range(N)]
                urgency_flags = [False] * N
                
                # Check if we've collected enough valid samples
                if len(samples) >= n_samples:
                    break            # Progress reporting
            if verbose and isinstance(pbar, tqdm):
                elapsed = time.time() - t_start
                samples_per_sec = len(samples) / elapsed if elapsed > 0 else 0
                eta = (n_samples - len(samples)) / samples_per_sec if samples_per_sec > 0 else 0
                n_reached = sum(reached)
                pbar.set_postfix({
                    "s/sample": f"{elapsed/max(len(samples),1):.2f}",
                    "ETA": f"{eta/60:.1f}m",
                    "ep_step": step_in_episode,
                    "reached": f"{n_reached}/{N}",
                    "ep": episode_count,
                    "valid": len(samples),
                    "buffer": len(episode_buffer),
                    "discarded": discarded_samples,
                })
        
        elapsed_total = time.time() - t_start
        actual_samples = len(samples)
        if verbose:
            print(f"\nCollection complete: {actual_samples} valid samples in {elapsed_total:.1f}s "
                  f"({elapsed_total/actual_samples:.2f}s/sample)")
            print(f"Discarded samples (from failed trajectories): {discarded_samples}")
            print(f"Total GPU forward passes: {gpu_calls} "
                  f"(avg {gpu_calls/actual_samples:.0f}/sample)")
            
            # Episode-level statistics
            print(f"\n--- Episode Statistics ---")
            print(f"Episodes: {episode_count} total")
            print(f"  All-reached (success): {all_reached_count}/{episode_count} "
                  f"({all_reached_count/max(episode_count,1)*100:.1f}%)")
            print(f"  Collision resets:      {collision_reset_count}/{episode_count} "
                  f"({collision_reset_count/max(episode_count,1)*100:.1f}%)")
            print(f"  Out-of-bounds resets:  {oob_reset_count}/{episode_count} "
                  f"({oob_reset_count/max(episode_count,1)*100:.1f}%)")
            print(f"  Timeout resets:        {timeout_reset_count}/{episode_count} "
                  f"({timeout_reset_count/max(episode_count,1)*100:.1f}%)")
            
            # Oracle rollout statistics (across all samples × all groups)
            oracle_collision_rate = total_oracle_collisions / max(total_oracle_groups_evaluated, 1)
            oracle_success_rate = 1.0 - oracle_collision_rate
            print(f"\n--- Oracle Rollout Statistics ---")
            print(f"Total group rollouts evaluated: {total_oracle_groups_evaluated} "
                  f"({actual_samples} samples × {len(self.groups)} groups)")
            print(f"  Collision rollouts: {total_oracle_collisions} "
                  f"({oracle_collision_rate*100:.1f}%)")
            print(f"  Non-collision rollouts (success): {total_oracle_groups_evaluated - total_oracle_collisions} "
                  f"({oracle_success_rate*100:.1f}%)")
            
            # Phase 2 group selection statistics
            print(f"\n--- Phase 2 Group Selection Statistics ---")
            print(f"Phase 2 selection mode: {phase2_mode}")
            if phase2_mode == "softmax":
                print(f"Temperature: {phase2_temperature}")
            print(f"Total Phase 2 selections: {actual_samples}")
            print(f"\nGroup selection breakdown:")
            print(f"{'Group':<8} {'Robots':<20} {'Count':<10} {'Percentage':<10}")
            print("-" * 50)
            for group_idx in sorted(phase2_group_counts.keys()):
                count = phase2_group_counts[group_idx]
                percentage = (count / actual_samples * 100) if actual_samples > 0 else 0
                robots_str = str(self.groups[group_idx])
                print(f"{group_idx:<8} {robots_str:<20} {count:<10} {percentage:>6.2f}%")
        
        data = {
            "samples": samples,
            "config": {
                "n_samples": actual_samples,
                "n_samples_requested": n_samples,
                "n_samples_discarded": discarded_samples,
                "embed_dim": CONFIG["embed_dim"],
                "n_robots": self.num_robots,
                "n_obstacles": self.num_obstacles,
                "n_groups": len(self.groups),
                "groups": self.groups,
                "horizon": self.horizon,
                "n_rollouts_per_group": self.n_rollouts_per_group,
                "collection_method": "simulation_rollout_batch_allreach",
                "scoring": "sync_newreach_laggard_urgency",
                "use_rotation_coupling": CONFIG.get("use_rotation_coupling", True),
                "k_reach": CONFIG.get("k_reach", 50.0),
                "k_progress": CONFIG.get("k_progress", 3.0),
                "k_rotation_progress": CONFIG.get("k_rotation_progress", 2.0),
                "k_urgency": CONFIG.get("k_urgency", 15.0),
                "k_sync": CONFIG.get("k_sync", 8.0),
                "urgency_lookback_window": CONFIG.get("urgency_lookback_window", 20),
                "urgency_stuck_threshold": CONFIG.get("urgency_stuck_threshold", 0.3),
                "goal_reach_threshold": goal_threshold,
                "max_steps_per_episode": max_steps,
                "episodes_total": episode_count,
                "episodes_all_reached": all_reached_count,
                "episodes_collision": collision_reset_count,
                "episodes_oob": oob_reset_count,
                "episodes_timeout": timeout_reset_count,
                "oracle_groups_evaluated": total_oracle_groups_evaluated,
                "oracle_collision_rollouts": total_oracle_collisions,
                "extra_features": [
                    "dist_to_goal", "clearance", "reached",
                    "heading_error", "urgency",
                    "var_dist_goal_global", "frac_reached_global",
                    "steps_elapsed_frac",
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
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("Oracle Data Collection for Group Switcher (All-Reach Synchronized)")
    print("=" * 70)
    print(f"Output path: {config['output_path']}")
    print(f"Number of samples: {config['n_samples']}")
    print(f"Number of robots: {config['n_robots']}")
    print(f"Embedding dimension: {config['embed_dim']}")
    print(f"Oracle horizon: {config['oracle_horizon']} steps")
    print(f"Rollouts per group: {config['n_rollouts_per_group']}")
    print(f"Goal reach threshold: {config.get('goal_reach_threshold', 0.3)}")
    print(f"Scoring components:")
    print(f"  - New-reach bonus: k={config.get('k_reach', 50.0)}")
    print(f"  - Laggard progress: k={config.get('k_progress', 3.0)}")
    print(f"  - Rotation progress: k={config.get('k_rotation_progress', 2.0)}")
    print(f"  - Urgency bonus: k={config.get('k_urgency', 15.0)}")
    print(f"  - Sync reward: k={config.get('k_sync', 3.0)} (only when ≥1 robot reached)")
    print(f"Episode reset: ALL robots reached OR max {config['max_steps_per_episode']} steps")
    # Group size info
    sizes_included = []
    if config["include_size_1"]: sizes_included.append("1")
    if config["include_size_2"]: sizes_included.append("2")
    if config["include_size_3"]: sizes_included.append("3")
    if config.get("include_size_4", False): sizes_included.append("4")
    if config.get("include_size_7", False): sizes_included.append("7")
    print(f"Group sizes included: {', '.join(sizes_included)}")
    if config.get("use_rotation_coupling", True):
        print(f"Rotation coupling: ON for groups with size > 3 (avg angular vel, min linear vel)")
    else:
        print(f"Rotation coupling: OFF (all groups use individual angular vel)")
    phase2_mode = config.get("phase2_selection", "random")
    if phase2_mode == "softmax":
        print(f"Phase 2 selection: softmax (temperature={config.get('phase2_temperature', 1.0)}, higher score → higher sample prob)")
    else:
        print(f"Phase 2 selection: random (uniform random group)")
    print("=" * 70 + "\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create simulation environment
    logger.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
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
        include_size_4=config.get("include_size_4", False),
        include_size_7=config.get("include_size_7", False),
    )
    
    logger.info(f"Candidate groups: {len(candidate_groups)} total")
    logger.info(f"  Size-1: {sum(1 for g in candidate_groups if len(g) == 1)}")
    logger.info(f"  Size-2: {sum(1 for g in candidate_groups if len(g) == 2)}")
    logger.info(f"  Size-3: {sum(1 for g in candidate_groups if len(g) == 3)}")
    logger.info(f"  Size-4: {sum(1 for g in candidate_groups if len(g) == 4)}")
    logger.info(f"  Size-7: {sum(1 for g in candidate_groups if len(g) == 7)}")
    if config.get("use_rotation_coupling", True):
        logger.info("  Rotation coupling: ON for groups with size > 3")
    
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
        n_samples=config["n_samples"],
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
    print(f"  samples[i]['h']: Tensor[{n_robots}, {embed_dim}]  (per-robot embeddings)")
    print(f"  samples[i]['groups']: List of {n_groups} groups")
    print(f"  samples[i]['group_scores']: Tensor[{n_groups}]  (oracle reward scores)")
    print(f"  samples[i]['attn_rr']: Tensor[{n_robots}, {n_robots}]")
    print(f"  samples[i]['attn_ro']: Tensor[{n_robots}, N_obs]")
    print(f"  samples[i]['extra']:")
    print(f"    Per-robot (N,):")
    print(f"      'dist_to_goal': Tensor[{n_robots}]  (per-robot distance to goal)")
    print(f"      'clearance': Tensor[{n_robots}]  (per-robot min obstacle clearance)")
    print(f"      'reached': Tensor[{n_robots}]  (per-robot sticky binary flag)")
    print(f"      'heading_error': Tensor[{n_robots}]  (per-robot |heading - goal_dir|)")
    print(f"    Global scalars (1,):")
    print(f"      'var_dist_goal_global': Tensor[1]  (distance variance across all robots)")
    print(f"      'frac_reached_global': Tensor[1]  (global completion fraction)")
    print(f"      'steps_elapsed_frac': Tensor[1]  (time pressure)")
    print(f"\n  13 per-group scalars (via GroupFeatureBuilder):")
    print(f"    Per-group: size_feat, coupling_mode, A_in, A_out, A_obs,")
    print(f"               mean_dist_goal_g, min_dist_goal_g, min_clearance_g,")
    print(f"               frac_reached_g, mean_heading_err_g")
    print(f"    Global:    var_dist_goal_global, frac_reached_global, steps_elapsed_frac")
    print(f"\n  Scoring: new-reach bonus + laggard progress + rotation progress + urgency bonus (N/A) + sync reward (conditional)")
    print(f"  Episode reset: all robots reached OR max {config['max_steps_per_episode']} steps")
    print(f"  Episodes: {data['config'].get('episodes_total', '?')} total, "
          f"{data['config'].get('episodes_all_reached', '?')} all-reached")
    
    print(f"\nTo train the switcher:")
    print(f"  python -m robot_nav.models.MARL.switcher.train_switcher")


if __name__ == "__main__":
    main()
