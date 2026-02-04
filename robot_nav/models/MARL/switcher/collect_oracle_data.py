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
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# =============================================================================
# Configuration Dictionary - Edit these values directly
# =============================================================================
CONFIG = {
    # Output configuration
    "output_path": "robot_nav/models/MARL/switcher/data/oracle_data.pt",
    
    # Data collection settings
    "n_samples": 10000,              # Number of samples to collect
    "n_robots": 6,                  # Number of robots
    "n_obstacles": 4,               # Number of obstacles
    "embed_dim": 256,               # Embedding dimension from GAT backbone
    "seed": 42,                     # Random seed for reproducibility
    
    # Oracle evaluation settings
    "oracle_horizon": 10,           # Number of steps to simulate forward for each group
    "n_rollouts_per_group": 1,      # Number of rollouts to average for each group score
    
    # Group generation settings
    "include_size_1": False,         # Include individual robots as candidates
    "include_size_2": True,         # Include pairs
    "include_size_3": True,         # Include triplets
    
    # Model configuration
    "state_dim": 11,
    "obstacle_state_dim": 4,
    
    # Pretrained model paths (decentralized TD3Obstacle policy)
    "decentralized_model_name": "TD3-MARL-obstacle-6robots_epoch2400",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots_v2",
    
    # Simulation settings
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle.yaml",
    "disable_plotting": True,
    "obstacle_proximity_threshold": 1.5,
    "max_steps_per_episode": 400,   # Reset episode after this many steps
    
    # Debug settings
    "debug_scoring": True,          # Enable detailed scoring debug output
    "debug_samples": 10,            # Number of samples to print detailed debug info
    
    # Interactive debug mode - pause after each oracle with rendering
    "interactive_debug": False,      # Enable interactive debug mode (pauses after each group eval)
    "interactive_samples": 3,       # Number of samples to run in interactive mode
}


# =============================================================================
# Debug Statistics Tracker
# =============================================================================
class ScoringDebugStats:
    """Track statistics of scoring components for debugging."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.collision_count = 0
        self.goal_count = 0
        self.progress_rewards = []
        self.evasion_rewards = []
        self.stuckness_penalties = []
        self.total_scores = []
        self.evasion_details = []  # List of dicts with breakdown
        
    def add_sample(self, collision, goal, progress, evasion, stuckness, total, evasion_detail=None):
        if collision:
            self.collision_count += 1
        if goal:
            self.goal_count += 1
        self.progress_rewards.append(progress)
        self.evasion_rewards.append(evasion)
        self.stuckness_penalties.append(stuckness)
        self.total_scores.append(total)
        if evasion_detail:
            self.evasion_details.append(evasion_detail)
    
    def print_summary(self, n_samples):
        print("\n" + "=" * 70)
        print("SCORING DEBUG SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal groups evaluated: {len(self.total_scores)}")
        print(f"Collisions: {self.collision_count} ({100*self.collision_count/max(1,len(self.total_scores)):.1f}%)")
        print(f"Goals reached: {self.goal_count} ({100*self.goal_count/max(1,len(self.total_scores)):.1f}%)")
        
        def stats(arr, name):
            if len(arr) == 0:
                return
            arr = np.array(arr)
            print(f"\n{name}:")
            print(f"  Mean: {arr.mean():.4f}")
            print(f"  Std:  {arr.std():.4f}")
            print(f"  Min:  {arr.min():.4f}")
            print(f"  Max:  {arr.max():.4f}")
            print(f"  |Mean|: {np.abs(arr).mean():.4f}  (absolute contribution)")
        
        stats(self.progress_rewards, "Progress Reward")
        stats(self.evasion_rewards, "Evasion Reward")
        stats(self.stuckness_penalties, "Stuckness Penalty")
        stats(self.total_scores, "Total Score")
        
        # Compute relative contributions (using absolute values)
        if len(self.total_scores) > 0:
            abs_progress = np.abs(self.progress_rewards).mean()
            abs_evasion = np.abs(self.evasion_rewards).mean()
            abs_stuckness = np.abs(self.stuckness_penalties).mean()
            total_contrib = abs_progress + abs_evasion + abs_stuckness + 1e-8
            
            print(f"\nRelative Contribution (by absolute mean):")
            print(f"  Progress:  {100*abs_progress/total_contrib:.1f}%")
            print(f"  Evasion:   {100*abs_evasion/total_contrib:.1f}%")
            print(f"  Stuckness: {100*abs_stuckness/total_contrib:.1f}%")
        
        # Evasion breakdown
        if len(self.evasion_details) > 0:
            robot_align = [d.get('robot_align', 0) for d in self.evasion_details]
            robot_dist = [d.get('robot_dist', 0) for d in self.evasion_details]
            obs_align = [d.get('obs_align', 0) for d in self.evasion_details]
            obs_dist = [d.get('obs_dist', 0) for d in self.evasion_details]
            
            print(f"\nEvasion Reward Breakdown:")
            print(f"  Robot alignment: mean={np.mean(robot_align):.4f}, std={np.std(robot_align):.4f}")
            print(f"  Robot distance:  mean={np.mean(robot_dist):.4f}, std={np.std(robot_dist):.4f}")
            print(f"  Obs alignment:   mean={np.mean(obs_align):.4f}, std={np.std(obs_align):.4f}")
            print(f"  Obs distance:    mean={np.mean(obs_dist):.4f}, std={np.std(obs_dist):.4f}")
        
        print("=" * 70 + "\n")


# Global debug tracker
DEBUG_STATS = ScoringDebugStats()

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
        robot_proximity_threshold: float = 1.5,
        obstacle_proximity_threshold: float = 1.5,
        return_details: bool = False,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Compute evasion reward for robots that rotate/move away from nearby entities.
        
        Rewards robots for:
        - Rotating away from nearby robots/obstacles (alignment improvement)
        - Increasing distance from nearby robots/obstacles (distance improvement)
        
        Only applies to entities within the proximity threshold.
        
        Args:
            group: Robot indices in the group
            initial_poses: Per-robot poses at start [[x, y, theta], ...]
            final_poses: Per-robot poses at end [[x, y, theta], ...]
            initial_obstacle_states: Obstacle states at start (N_obs, 4) [x, y, vx, vy]
            final_obstacle_states: Obstacle states at end
            robot_proximity_threshold: Only consider robots within this distance
            obstacle_proximity_threshold: Only consider obstacles within this distance
            return_details: If True, return breakdown of evasion components
            
        Returns:
            evasion_score: Reward for evasive maneuvers (higher = better)
            details: Optional dict with breakdown (if return_details=True)
        """
        evasion_score = 0.0
        k_align = 5.0   # Weight for alignment improvement
        k_dist = 3.0    # Weight for distance improvement
        
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
                
                # Initial distance and angle to robot j
                xj_init, yj_init, _ = initial_poses[j]
                dx_init = xj_init - xi_init
                dy_init = yj_init - yi_init
                dist_init = np.sqrt(dx_init**2 + dy_init**2)
                
                # Only consider robots within threshold
                if dist_init > robot_proximity_threshold:
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
                dist_final = np.sqrt(dx_final**2 + dy_final**2)
                angle_to_j_final = np.arctan2(dy_final, dx_final)
                alignment_final = np.cos(theta_i_final - angle_to_j_final)
                
                # Improvement scores (positive = good)
                # Alignment improvement: went from pointing at (1) to pointing away (-1)
                alignment_improvement = alignment_init - alignment_final
                # Distance improvement: increased distance
                dist_improvement = dist_final - dist_init
                
                # Urgency weight: closer initial distance = more important to evade
                urgency = max(0, robot_proximity_threshold - dist_init) / robot_proximity_threshold
                
                robot_align_contrib = urgency * k_align * alignment_improvement
                robot_dist_contrib = urgency * k_dist * dist_improvement
                robot_align_total += robot_align_contrib
                robot_dist_total += robot_dist_contrib
                evasion_score += robot_align_contrib + robot_dist_contrib
            
            # === Robot-Obstacle Evasion ===
            for obs_idx in range(self.num_obstacles):
                # Initial obstacle position
                ox_init = initial_obstacle_states[obs_idx, 0]
                oy_init = initial_obstacle_states[obs_idx, 1]
                
                dx_init = ox_init - xi_init
                dy_init = oy_init - yi_init
                dist_init = np.sqrt(dx_init**2 + dy_init**2)
                
                # Only consider obstacles within threshold
                if dist_init > obstacle_proximity_threshold:
                    continue
                
                angle_to_obs_init = np.arctan2(dy_init, dx_init)
                alignment_init = np.cos(theta_i_init - angle_to_obs_init)
                
                # Final obstacle position
                ox_final = final_obstacle_states[obs_idx, 0]
                oy_final = final_obstacle_states[obs_idx, 1]
                
                dx_final = ox_final - xi_final
                dy_final = oy_final - yi_final
                dist_final = np.sqrt(dx_final**2 + dy_final**2)
                angle_to_obs_final = np.arctan2(dy_final, dx_final)
                alignment_final = np.cos(theta_i_final - angle_to_obs_final)
                
                # Improvement scores
                alignment_improvement = alignment_init - alignment_final
                dist_improvement = dist_final - dist_init
                
                # Urgency weight
                urgency = max(0, obstacle_proximity_threshold - dist_init) / obstacle_proximity_threshold
                
                obs_align_contrib = urgency * k_align * alignment_improvement
                obs_dist_contrib = urgency * k_dist * dist_improvement
                obs_align_total += obs_align_contrib
                obs_dist_total += obs_dist_contrib
                evasion_score += obs_align_contrib + obs_dist_contrib
        
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
        had_goal: bool = False,
    ) -> float:
        """
        Compute penalty for groups that result in very low movement (stuckness).
        
        This discourages the switcher from selecting groups that don't make progress.
        
        Args:
            group: Robot indices in the group
            initial_poses: Per-robot poses at start [[x, y, theta], ...]
            final_poses: Per-robot poses at end [[x, y, theta], ...]
            min_displacement_threshold: Minimum expected displacement over horizon
            had_goal: If True, don't penalize (robot reached goal, stopping is OK)
            
        Returns:
            stuckness_penalty: Negative value if group is stuck (lower = worse)
        """
        if had_goal:
            # Don't penalize if goal was reached
            return 0.0
        
        k_stuck = 20.0
        
        # Compute average displacement of robots in the group
        total_displacement = 0.0
        for i in group:
            xi_init, yi_init, _ = initial_poses[i]
            xi_final, yi_final, _ = final_poses[i]
            displacement = np.sqrt((xi_final - xi_init)**2 + (yi_final - yi_init)**2)
            total_displacement += displacement
        
        avg_displacement = total_displacement / len(group)
        
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
        had_goal: bool,
        robot_proximity_threshold: float = 1.5,
        obstacle_proximity_threshold: float = 1.5,
        min_displacement_threshold: float = 0.2,
        debug: bool = False,
    ) -> float:
        """
        Compute trajectory-based score for an oracle rollout.
        
        This evaluates the entire H-step trajectory holistically based on
        start and end states.
        
        Scoring components:
        1. Collision penalty: -50 if any collision occurred
        2. Goal bonus: +50 if any robot in group reached goal
        3. Progress reward: 10.0 * sum(initial_dist - final_dist) for robots in group
        4. Evasion reward: Reward for rotating/moving away from nearby entities
        5. Stuckness penalty: Penalty for groups with very low displacement
        
        Args:
            group: Robot indices in the group
            initial_poses: Per-robot poses at start [[x, y, theta], ...]
            final_poses: Per-robot poses at end [[x, y, theta], ...]
            initial_distances: Per-robot distances to goal at step 0
            final_distances: Per-robot distances to goal at final step
            initial_obstacle_states: Obstacle states at start
            final_obstacle_states: Obstacle states at end
            had_collision: Whether collision occurred during trajectory
            had_goal: Whether any robot in group reached goal
            robot_proximity_threshold: Threshold for evasion reward (robots)
            obstacle_proximity_threshold: Threshold for evasion reward (obstacles)
            min_displacement_threshold: Minimum displacement to avoid stuckness penalty
            debug: If True, track debug statistics
            
        Returns:
            score: Trajectory score (higher = better)
        """
        # 1. Collision penalty
        if had_collision:
            if debug:
                DEBUG_STATS.add_sample(
                    collision=True, goal=False, progress=0, evasion=0, 
                    stuckness=0, total=-50.0, evasion_detail=None
                )
            return -50.0
        
        score = 0.0
        
        # 2. Goal bonus
        if had_goal:
            score += 50.0
        
        # 3. Progress reward: sum of (initial_dist - final_dist) for robots in group
        k_progress = 10.0
        progress_reward = 0.0
        for i in group:
            progress = initial_distances[i] - final_distances[i]
            progress_reward += k_progress * progress
        score += progress_reward
        
        # 4. Evasion reward: reward for rotating/moving away from nearby entities
        evasion_reward, evasion_detail = self.compute_evasion_reward(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            initial_obstacle_states=initial_obstacle_states,
            final_obstacle_states=final_obstacle_states,
            robot_proximity_threshold=robot_proximity_threshold,
            obstacle_proximity_threshold=obstacle_proximity_threshold,
            return_details=debug,
        )
        score += evasion_reward
        
        # 5. Stuckness penalty: penalize groups with very low displacement
        stuckness_penalty = self.compute_stuckness_penalty(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            min_displacement_threshold=min_displacement_threshold,
            had_goal=had_goal,
        )
        score += stuckness_penalty
        
        # Track debug statistics
        if debug:
            DEBUG_STATS.add_sample(
                collision=False,
                goal=had_goal,
                progress=progress_reward,
                evasion=evasion_reward,
                stuckness=stuckness_penalty,
                total=score,
                evasion_detail=evasion_detail,
            )
        
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
        debug: bool = False,
        interactive: bool = False,
    ) -> Tuple[float, bool]:
        """
        Evaluate a group by simulating forward H steps (single rollout).
        
        Uses trajectory-based scoring that evaluates the entire rollout:
        - Collision penalty if any collision occurred
        - Goal bonus if goal reached
        - Progress reward based on initial vs final distance
        - Evasion reward for rotating/moving away from nearby entities
        - Stuckness penalty for groups with very low displacement
        
        Args:
            group: Robot indices in the group.
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after rollout.
            debug: If True, track debug statistics.
            interactive: If True, render simulation and print detailed breakdown.
            
        Returns:
            Tuple of (trajectory_score, had_collision).
        """
        had_collision = False
        had_goal = False
        
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
        
        if interactive:
            print(f"\n{'='*60}")
            print(f"ORACLE ROLLOUT: Group {group}")
            print(f"{'='*60}")
            print(f"Initial poses: {[f'R{i}:({p[0]:.2f},{p[1]:.2f},θ={p[2]:.2f})' for i, p in enumerate(initial_poses) if i in group]}")
            print(f"Initial distances to goal: {[f'R{i}:{d:.2f}' for i, d in enumerate(initial_distances) if i in group]}")
        
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
            
            # Render if interactive
            if interactive:
                self.sim.render()
            
            # Check for goal reached by any robot in group
            for i in group:
                if curr_goal[i]:
                    had_goal = True
            
            # Check for collision - end rollout early if collision
            if any(curr_collision[i] for i in group):
                had_collision = True
                if interactive:
                    print(f"  Step {step}: COLLISION detected!")
                break
            
            # Check for out of bounds
            if outside_of_bounds(curr_poses, self.sim):
                had_collision = True
                if interactive:
                    print(f"  Step {step}: OUT OF BOUNDS!")
                break
        
        # Final state after trajectory
        final_poses = [p.copy() for p in curr_poses]
        final_distances = curr_distance.copy()
        final_obstacle_states = curr_obstacle_states.copy()
        
        # Compute detailed score breakdown for interactive mode
        if interactive:
            self._print_interactive_score_breakdown(
                group=group,
                initial_poses=initial_poses,
                final_poses=final_poses,
                initial_distances=initial_distances,
                final_distances=final_distances,
                initial_obstacle_states=initial_obstacle_states,
                final_obstacle_states=final_obstacle_states,
                had_collision=had_collision,
                had_goal=had_goal,
            )
        
        # Compute trajectory score using start and end states
        trajectory_score = self.compute_trajectory_score(
            group=group,
            initial_poses=initial_poses,
            final_poses=final_poses,
            initial_distances=initial_distances,
            final_distances=final_distances,
            initial_obstacle_states=initial_obstacle_states,
            final_obstacle_states=final_obstacle_states,
            had_collision=had_collision,
            had_goal=had_goal,
            robot_proximity_threshold=1.5,
            obstacle_proximity_threshold=self.sim.obstacle_proximity_threshold,
            min_displacement_threshold=0.2,
            debug=debug,
        )
        
        if interactive:
            print(f"\n  >>> TOTAL SCORE: {trajectory_score:.4f}")
            input("  Press Enter to continue to next group...")
        
        # Restore simulation to original state
        snapshot.restore_to_sim(self.sim)
        
        # Re-render after restore if interactive
        if interactive:
            self.sim.render()
        
        return trajectory_score, had_collision
    
    def _print_interactive_score_breakdown(
        self,
        group: List[int],
        initial_poses: List[List[float]],
        final_poses: List[List[float]],
        initial_distances: List[float],
        final_distances: List[float],
        initial_obstacle_states: np.ndarray,
        final_obstacle_states: np.ndarray,
        had_collision: bool,
        had_goal: bool,
    ):
        """Print detailed score breakdown for interactive debugging."""
        print(f"\n--- Score Breakdown for Group {group} ---")
        
        # Collision
        if had_collision:
            print(f"  [COLLISION] Score = -50.0 (early termination)")
            return
        
        # Goal bonus
        goal_bonus = 50.0 if had_goal else 0.0
        print(f"  [GOAL]      Reached: {had_goal}, Bonus: +{goal_bonus:.2f}")
        
        # Progress reward
        k_progress = 10.0
        progress_reward = 0.0
        print(f"  [PROGRESS]  (k={k_progress})")
        for i in group:
            progress = initial_distances[i] - final_distances[i]
            contrib = k_progress * progress
            progress_reward += contrib
            print(f"              R{i}: {initial_distances[i]:.3f} -> {final_distances[i]:.3f}, "
                  f"Δ={progress:.3f}, contrib={contrib:.3f}")
        print(f"              Total progress reward: {progress_reward:.4f}")
        
        # Evasion reward breakdown
        k_align = 5.0
        k_dist = 3.0
        robot_prox_thresh = 1.5
        obs_prox_thresh = self.sim.obstacle_proximity_threshold
        
        print(f"  [EVASION]   (k_align={k_align}, k_dist={k_dist})")
        
        robot_align_total = 0.0
        robot_dist_total = 0.0
        obs_align_total = 0.0
        obs_dist_total = 0.0
        
        for i in group:
            xi_init, yi_init, theta_i_init = initial_poses[i]
            xi_final, yi_final, theta_i_final = final_poses[i]
            
            # Robot-robot evasion
            for j in range(self.num_robots):
                if i == j:
                    continue
                xj_init, yj_init, _ = initial_poses[j]
                dist_init = np.sqrt((xj_init - xi_init)**2 + (yj_init - yi_init)**2)
                
                if dist_init <= robot_prox_thresh:
                    angle_to_j_init = np.arctan2(yj_init - yi_init, xj_init - xi_init)
                    alignment_init = np.cos(theta_i_init - angle_to_j_init)
                    
                    xj_final, yj_final, _ = final_poses[j]
                    dist_final = np.sqrt((xj_final - xi_final)**2 + (yj_final - yi_final)**2)
                    angle_to_j_final = np.arctan2(yj_final - yi_final, xj_final - xi_final)
                    alignment_final = np.cos(theta_i_final - angle_to_j_final)
                    
                    alignment_improvement = alignment_init - alignment_final
                    dist_improvement = dist_final - dist_init
                    urgency = max(0, robot_prox_thresh - dist_init) / robot_prox_thresh
                    
                    align_contrib = urgency * k_align * alignment_improvement
                    dist_contrib = urgency * k_dist * dist_improvement
                    robot_align_total += align_contrib
                    robot_dist_total += dist_contrib
                    
                    print(f"              R{i}->R{j}: dist={dist_init:.2f}, urgency={urgency:.2f}")
                    print(f"                align: {alignment_init:.3f}->{alignment_final:.3f}, Δ={alignment_improvement:.3f}, contrib={align_contrib:.3f}")
                    print(f"                dist:  {dist_init:.3f}->{dist_final:.3f}, Δ={dist_improvement:.3f}, contrib={dist_contrib:.3f}")
            
            # Robot-obstacle evasion
            for obs_idx in range(self.num_obstacles):
                ox_init = initial_obstacle_states[obs_idx, 0]
                oy_init = initial_obstacle_states[obs_idx, 1]
                dist_init = np.sqrt((ox_init - xi_init)**2 + (oy_init - yi_init)**2)
                
                if dist_init <= obs_prox_thresh:
                    angle_to_obs_init = np.arctan2(oy_init - yi_init, ox_init - xi_init)
                    alignment_init = np.cos(theta_i_init - angle_to_obs_init)
                    
                    ox_final = final_obstacle_states[obs_idx, 0]
                    oy_final = final_obstacle_states[obs_idx, 1]
                    dist_final = np.sqrt((ox_final - xi_final)**2 + (oy_final - yi_final)**2)
                    angle_to_obs_final = np.arctan2(oy_final - yi_final, ox_final - xi_final)
                    alignment_final = np.cos(theta_i_final - angle_to_obs_final)
                    
                    alignment_improvement = alignment_init - alignment_final
                    dist_improvement = dist_final - dist_init
                    urgency = max(0, obs_prox_thresh - dist_init) / obs_prox_thresh
                    
                    align_contrib = urgency * k_align * alignment_improvement
                    dist_contrib = urgency * k_dist * dist_improvement
                    obs_align_total += align_contrib
                    obs_dist_total += dist_contrib
                    
                    print(f"              R{i}->Obs{obs_idx}: dist={dist_init:.2f}, urgency={urgency:.2f}")
                    print(f"                align: {alignment_init:.3f}->{alignment_final:.3f}, Δ={alignment_improvement:.3f}, contrib={align_contrib:.3f}")
                    print(f"                dist:  {dist_init:.3f}->{dist_final:.3f}, Δ={dist_improvement:.3f}, contrib={dist_contrib:.3f}")
        
        evasion_total = robot_align_total + robot_dist_total + obs_align_total + obs_dist_total
        print(f"              Robot align: {robot_align_total:.4f}, Robot dist: {robot_dist_total:.4f}")
        print(f"              Obs align: {obs_align_total:.4f}, Obs dist: {obs_dist_total:.4f}")
        print(f"              Total evasion reward: {evasion_total:.4f}")
        
        # Stuckness penalty
        k_stuck = 20.0
        min_disp_thresh = 0.2
        total_displacement = 0.0
        print(f"  [STUCKNESS] (k={k_stuck}, thresh={min_disp_thresh})")
        for i in group:
            xi_init, yi_init, _ = initial_poses[i]
            xi_final, yi_final, _ = final_poses[i]
            disp = np.sqrt((xi_final - xi_init)**2 + (yi_final - yi_init)**2)
            total_displacement += disp
            print(f"              R{i}: displacement={disp:.4f}")
        
        avg_disp = total_displacement / len(group)
        if had_goal:
            stuckness_penalty = 0.0
            print(f"              Avg displacement: {avg_disp:.4f} (no penalty - goal reached)")
        elif avg_disp < min_disp_thresh:
            stuckness_penalty = -k_stuck * (min_disp_thresh - avg_disp)
            print(f"              Avg displacement: {avg_disp:.4f} < {min_disp_thresh}, penalty={stuckness_penalty:.4f}")
        else:
            stuckness_penalty = 0.0
            print(f"              Avg displacement: {avg_disp:.4f} >= {min_disp_thresh}, no penalty")
        
        # Summary
        total = goal_bonus + progress_reward + evasion_total + stuckness_penalty
        print(f"\n  --- SUMMARY ---")
        print(f"  Goal bonus:      {goal_bonus:+.4f}")
        print(f"  Progress reward: {progress_reward:+.4f}")
        print(f"  Evasion reward:  {evasion_total:+.4f}")
        print(f"  Stuckness pen:   {stuckness_penalty:+.4f}")
        print(f"  --------------------------")
        print(f"  TOTAL:           {total:+.4f}")
    
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
        debug: bool = False,
        interactive: bool = False,
    ) -> float:
        """
        Evaluate a group by averaging over n_rollouts.
        
        Args:
            group: List of robot indices in the group
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after each rollout.
            debug: If True, track debug statistics.
            interactive: If True, render and pause for user inspection.
            
        Returns:
            score: Average cumulative reward across rollouts (higher = better)
        """
        total_reward = 0.0
        
        for rollout_idx in range(self.n_rollouts_per_group):
            reward, _ = self._evaluate_group_once(
                group, poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states, snapshot, debug=debug,
                interactive=interactive,
            )
            total_reward += reward
        
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
    ) -> Dict[str, torch.Tensor]:
        """
        Get extra per-robot features from environment state.
        
        Args:
            poses: Per-robot poses [[x, y, theta], ...]
            distance: Per-robot distances to goal
            goal_positions: Per-robot goals [[gx, gy], ...]
            
        Returns:
            extra: Dict with "dist_to_goal", "clearance", etc.
        """
        # Distance to goal
        dist_to_goal = torch.tensor(distance, dtype=torch.float32)
        
        # Minimum clearance to obstacles
        clearances = []
        for i in range(self.num_robots):
            min_clearance = self.sim.get_min_obstacle_clearance(i)
            clearances.append(min_clearance)
        clearance = torch.tensor(clearances, dtype=torch.float32)
        
        return {
            "dist_to_goal": dist_to_goal,
            "clearance": clearance,
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
        debug: bool = False,
        interactive: bool = False,
    ) -> Dict:
        """
        Collect one sample of oracle data at the current simulation state.
        
        1. Get robot embeddings and attention from policy
        2. For each group, run rollouts and compute average score
        3. Return sample dict
        
        Args:
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state from sim.step() or sim.reset()
            scenario_id: Optional identifier for this sample
            debug: If True, track debug statistics
            interactive: If True, render and pause after each group evaluation
            
        Returns:
            Sample dictionary compatible with train_switcher.py
        """
        # Take snapshot before any rollouts
        snapshot = SimulationSnapshot.from_sim(self.sim)
        
        # Prepare robot observations using the policy's prepare_state method
        robot_state, _ = self.policy.prepare_state(
            poses, distance, cos, sin, collision, action, goal_positions
        )
        robot_obs = np.array(robot_state)
        
        # Get embeddings and attention
        h, attn_rr, attn_ro = self._get_embeddings_and_attention(robot_obs, obstacle_states)
        
        # Get extra features
        extra = self._get_extra_features(poses, distance, goal_positions)
        
        if interactive:
            print(f"\n{'#'*70}")
            print(f"# SAMPLE {scenario_id}: Evaluating {len(self.groups)} groups")
            print(f"# Robot positions: {[f'R{i}:({p[0]:.1f},{p[1]:.1f})' for i, p in enumerate(poses)]}")
            print(f"# Obstacle positions: {[f'O{i}:({obstacle_states[i,0]:.1f},{obstacle_states[i,1]:.1f})' for i in range(self.num_obstacles)]}")
            print(f"{'#'*70}")
            # Render initial state
            self.sim.render()
            input("Press Enter to start evaluating groups...")
        
        # Evaluate each group with rollouts
        group_scores = []
        
        for group in self.groups:
            score = self._evaluate_group(
                group, poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states, snapshot, debug=debug,
                interactive=interactive,
            )
            group_scores.append(score)
        
        group_scores = torch.tensor(group_scores, dtype=torch.float32)
        
        return {
            "h": h.cpu(),
            "groups": self.groups,
            "group_scores": group_scores,
            "attn_rr": attn_rr.cpu(),
            "attn_ro": attn_ro.cpu(),
            "extra": {k: v.cpu() for k, v in extra.items()},
            "metadata": {
                "scenario_id": scenario_id,
            },
        }
    
    def collect_dataset(
        self,
        n_samples: int,
        save_path: Optional[str] = None,
        verbose: bool = True,
        debug_scoring: bool = False,
        debug_samples: int = 5,
    ) -> Dict:
        """
        Collect a full dataset of oracle samples by running episodes.
        
        Args:
            n_samples: Number of samples to collect
            save_path: Path to save the dataset (optional)
            verbose: Print progress bar
            debug_scoring: If True, track and print debug statistics
            debug_samples: Number of samples to collect debug stats for
            
        Returns:
            data: Dataset dictionary
        """
        samples = []
        
        # Check for interactive mode
        interactive_debug = CONFIG.get("interactive_debug", False)
        interactive_samples = CONFIG.get("interactive_samples", 3)
        
        # Reset debug stats
        if debug_scoring:
            DEBUG_STATS.reset()
        
        pbar = tqdm(range(n_samples), desc="Collecting oracle data") if verbose else range(n_samples)
        
        # Reset environment to start
        (
            poses, distance, cos, sin, collision, goals,
            action, reward, positions, goal_positions, obstacle_states
        ) = self.sim.reset()
        
        step_in_episode = 0
        max_steps = CONFIG.get("max_steps_per_episode", 500)
        
        for i in pbar:
            # Enable debug for first N samples
            enable_debug = debug_scoring and (i < debug_samples)
            # Enable interactive mode for first few samples
            enable_interactive = interactive_debug and (i < interactive_samples)
            
            # Collect sample at current state
            sample = self.collect_sample(
                poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states,
                scenario_id=i,
                debug=enable_debug,
                interactive=enable_interactive,
            )
            samples.append(sample)
            
            # Print sample summary after interactive session
            if enable_interactive:
                scores = sample["group_scores"]
                best_idx = scores.argmax().item()
                worst_idx = scores.argmin().item()
                print(f"\n{'='*70}")
                print(f"SAMPLE {i} COMPLETE - Summary")
                print(f"{'='*70}")
                print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
                print(f"  Best group: {self.groups[best_idx]} (score={scores[best_idx]:.2f})")
                print(f"  Worst group: {self.groups[worst_idx]} (score={scores[worst_idx]:.2f})")
                print(f"  Mean score: {scores.mean():.2f}, Std: {scores.std():.2f}")
                
                # Print top 5 groups
                sorted_indices = scores.argsort(descending=True)
                print(f"\n  Top 5 groups:")
                for rank, idx in enumerate(sorted_indices[:5]):
                    print(f"    {rank+1}. {self.groups[idx]} -> score={scores[idx]:.2f}")
                
                input("\nPress Enter to continue to next sample...")
            
            # Print detailed debug info for first few samples (non-interactive)
            elif enable_debug and verbose:
                scores = sample["group_scores"]
                best_idx = scores.argmax().item()
                worst_idx = scores.argmin().item()
                print(f"\n--- Sample {i} Debug ---")
                print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
                print(f"  Best group: {self.groups[best_idx]} (score={scores[best_idx]:.2f})")
                print(f"  Worst group: {self.groups[worst_idx]} (score={scores[worst_idx]:.2f})")
                print(f"  Mean score: {scores.mean():.2f}, Std: {scores.std():.2f}")
            
            # Take action from a randomly selected group to advance simulation
            # This better mimics how the switcher will operate during evaluation
            robot_state, _ = self.policy.prepare_state(
                poses, distance, cos, sin, collision, action, goal_positions
            )
            robot_obs = np.array(robot_state)
            
            # Randomly select a group and use its coupled action
            random_group = random.choice(self.groups)
            scaled_action = self.get_action_for_group(
                robot_obs, obstacle_states, random_group
            )
            
            # Step simulation
            (
                poses, distance, cos, sin, collision, goals,
                action, reward, positions, goal_positions, obstacle_states
            ) = self.sim.step(scaled_action, None, None)
            
            step_in_episode += 1
            
            # Check for episode reset conditions
            should_reset = (
                any(collision) or 
                step_in_episode >= max_steps or
                outside_of_bounds(poses, self.sim)
            )
            
            if should_reset:
                # Full reset: all robots and randomize obstacles
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states
                ) = self.sim.reset(random_obstacles=True)
                step_in_episode = 0
        
        # Print debug summary
        if debug_scoring and verbose:
            DEBUG_STATS.print_summary(debug_samples)
        
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
                "collection_method": "simulation_rollout",
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
    
    # Check for interactive mode
    interactive_debug = config.get("interactive_debug", False)
    interactive_samples = config.get("interactive_samples", 3)
    
    print("=" * 70)
    print("Oracle Data Collection for Group Switcher")
    print("=" * 70)
    print(f"Output path: {config['output_path']}")
    print(f"Number of samples: {config['n_samples']}")
    print(f"Number of robots: {config['n_robots']}")
    print(f"Embedding dimension: {config['embed_dim']}")
    print(f"Oracle horizon: {config['oracle_horizon']} steps")
    print(f"Rollouts per group: {config['n_rollouts_per_group']}")
    if interactive_debug:
        print(f"\n*** INTERACTIVE MODE ENABLED ***")
        print(f"    Will pause and render for first {interactive_samples} samples")
        print(f"    Press Enter to advance through group evaluations")
    print("=" * 70 + "\n")
    

    # Real data collection using simulation
    print("Collecting REAL oracle data via simulation rollouts...")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create simulation environment
    # In interactive mode, enable plotting for visualization
    disable_plotting = config["disable_plotting"]
    if interactive_debug:
        disable_plotting = False  # Enable rendering for interactive debug
        logger.info("Interactive mode: enabling simulation rendering")
    
    logger.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=disable_plotting,
        reward_phase=5,
        per_robot_goal_reset=True,
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )
    logger.info(f"Environment: {sim.num_robots} robots, {sim.num_obstacles} obstacles")
    
    # Load decentralized policy (TD3Obstacle)
    # This policy provides:
    # - Robot embeddings from GAT encoder
    # - Attention weights (hard_weights_rr, hard_weights_ro)
    # - Per-robot actions (we average linear velocities for coupled groups)
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
    
    # Collect data with debug mode from config
    debug_scoring = config.get("debug_scoring", False)
    debug_samples = config.get("debug_samples", 5)
    
    if debug_scoring:
        print(f"\n[DEBUG MODE] Collecting detailed scoring stats for first {debug_samples} samples...")
    
    data = collector.collect_dataset(
        n_samples=config["n_samples"],
        save_path=None,  # We'll save below
        verbose=True,
        debug_scoring=debug_scoring,
        debug_samples=debug_samples,
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
    print(f"  samples[i]['extra']: {{'dist_to_goal': Tensor[{n_robots}], 'clearance': Tensor[{n_robots}]}}")
    
    print(f"\nTo train the switcher:")
    print(f"  python -m robot_nav.models.MARL.switcher.train_switcher")


if __name__ == "__main__":
    main()
