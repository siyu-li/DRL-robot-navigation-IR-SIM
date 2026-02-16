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
    python -m robot_nav.models.MARL.switcher.test_switcher_v2

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
from robot_nav.models.MARL.switcher import (
    GroupFeatureBuilder,
    GroupSwitcher,
    generate_all_groups,
    filter_groups_by_size,
)
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
# Suppress IRSim warnings - irsim uses loguru, not standard logging
from loguru import logger
logger.disable("irsim")

# =============================================================================
# Configuration Dictionary - Edit these values directly
# =============================================================================
CONFIG = {
    # Selection mode: "switcher" or "random"
    "selection_mode": "random",  # Change to "random" for baseline comparison

    # Top-k selection: randomly select from top k groups instead of always best
    # Set to 1 for deterministic (always best), >1 for stochastic selection
    "top_k_selection": 10,

    # Random seed for reproducibility (set to same value for fair comparison)
    "seed": 42,

    # Switcher model configuration
    "switcher_checkpoint": "robot_nav/models/MARL/switcher/runs/switcher/best.pt",

    # Decentralized model configuration (used for all action generation)
    # "decentralized_model_name": "TD3-MARL-obstacle-14robots",
    # "decentralized_model_name": "TD3-MARL-obstacle-6robots_epoch2400",
    # "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Feb.8_obstacle_14robot_transfer",
    # "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots_v2",
    # Pretrained model paths (decentralized TD3Obstacle policy)
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-gpu_epoch800",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Feb.10_obstacle_14robot_transfer_gpu",

    # Test configuration
    "test_episodes": 90,
    "max_steps_per_episode": 1000,
    "disable_plotting": True,

    # Group selection interval (re-select group every N steps)
    "selection_interval": 10,

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
    "extra_features": ["dist_to_goal", "clearance"],
    "extra_aggregations": ["mean", "min"],

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
) -> List[List[int]]:
    """
    Generate all candidate groups using binary allocation method.

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
    """

    # Episode-level metrics
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    episode_outcomes: List[str] = field(default_factory=list)
    # "success" = all robots reached goal, no collisions
    # "collision" = at least one collision or out-of-bounds occurred
    # "timeout" = max_steps reached without all robots finishing

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
        path_lengths: Optional[List[float]] = None,
    ):
        """Record episode-level metrics."""
        self.episode_rewards.append(float(total_reward))
        self.episode_steps.append(int(steps))
        self.episode_outcomes.append(outcome)
        if outcome == "success" and path_lengths is not None:
            self.success_episode_path_lengths.append(list(path_lengths))

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        num_episodes = len(self.episode_rewards)
        total_executions = sum(self.executed_size_counts.values())

        # Episode statistics
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0.0

        # Episode-level success / collision / timeout counts
        success_count = sum(1 for o in self.episode_outcomes if o == "success")
        collision_count = sum(1 for o in self.episode_outcomes if o == "collision")
        timeout_count = sum(1 for o in self.episode_outcomes if o == "timeout")

        success_rate = success_count / max(num_episodes, 1)
        collision_rate = collision_count / max(num_episodes, 1)
        timeout_rate = timeout_count / max(num_episodes, 1)

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
        for size in [1, 2, 3]:
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
        for size in [1, 2, 3]:
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
            "avg_steps": avg_steps,
            "success_count": success_count,
            "collision_count": collision_count,
            "timeout_count": timeout_count,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
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
        top_k: int = 1,
    ):
        self.switcher = switcher.to(device)
        self.switcher.eval()
        self.feature_builder = feature_builder
        self.policy = policy
        self.groups = groups
        self.device = device
        self.top_k = max(1, min(top_k, len(groups)))

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
        """
        h, attn_rr, attn_ro = self.get_embeddings_and_attention(robot_obs, obstacle_obs)
        extra = self.get_extra_features(distance, sim)

        X = self.feature_builder(
            h=h,
            groups=self.groups,
            h_glob=None,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            extra=extra,
        )

        X = X.to(self.device)
        logits = self.switcher(X)

        if self.top_k == 1:
            selected_idx = logits.argmax().item()
        else:
            _, top_k_indices = torch.topk(logits, k=self.top_k)
            random_idx = random.randint(0, self.top_k - 1)
            selected_idx = top_k_indices[random_idx].item()

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
) -> List[List[float]]:
    """
    Get action for a specific group using the decentralized policy.

    All robots in the group get coupled actions (velocity-matched).
    Robots NOT in the group get [0, 0].
    Robots that have already reached their goal continue to receive
    policy actions — the learned policy will naturally keep them near the goal.

    Args:
        policy: TD3Obstacle decentralized policy.
        robot_obs: Robot observations, shape (num_robots, state_dim).
        obstacle_obs: Obstacle observations, shape (num_obstacles, obs_dim).
        group: List of robot indices in the active group.
        num_robots: Total number of robots.

    Returns:
        Actions for all robots, shape (num_robots, 2).
    """
    action, _ = policy.get_action(robot_obs, obstacle_obs, add_noise=False)

    # Get scaled linear velocities for robots in the group
    group_set = set(group)
    scaled_lin_vels = []
    for idx in group:
        scaled_lin_vel = (action[idx][0] + 1) / 4  # [-1,1] -> [0,0.5]
        scaled_lin_vels.append(scaled_lin_vel)

    v_coupled = min(scaled_lin_vels)

    a_out = []
    for i in range(num_robots):
        if i in group_set:
            a_out.append([v_coupled, action[i][1]])
        else:
            # Not in the active group
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
    seed: int = 42,
    verbose: bool = True,
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
    """
    stats = TestStatistics()
    num_robots = sim.num_robots

    pbar = (
        tqdm(range(num_episodes), desc=f"Testing ({selection_mode})")
        if verbose
        else range(num_episodes)
    )

    for episode in pbar:
        # ------------------------------------------------------------------
        # Deterministic per-episode seed
        # ------------------------------------------------------------------
        episode_seed = seed + episode
        set_global_seed(episode_seed)

        # Reset environment (all robots + random obstacles)
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

        while steps < max_steps:
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
                        robot_obs, obstacle_states, distance, sim
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
        # Determine episode outcome
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
            path_lengths=path_lengths,
        )

        if verbose and isinstance(pbar, tqdm):
            running_success = sum(1 for o in stats.episode_outcomes if o == "success")
            running_collision = sum(1 for o in stats.episode_outcomes if o == "collision")
            running_timeout = sum(1 for o in stats.episode_outcomes if o == "timeout")
            n = len(stats.episode_outcomes)

            pbar.set_postfix({
                "reward": f"{episode_reward:.1f}",
                "S": f"{running_success}/{n}",
                "C": f"{running_collision}/{n}",
                "T": f"{running_timeout}/{n}",
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
        include_size_1=True,
        include_size_2=True,
        include_size_3=True,
    )

    logger.info(f"Candidate groups: {len(groups)} total")
    logger.info(f"  Size-1: {sum(1 for g in groups if len(g) == 1)}")
    logger.info(f"  Size-2: {sum(1 for g in groups if len(g) == 2)}")
    logger.info(f"  Size-3: {sum(1 for g in groups if len(g) == 3)}")

    # ------------------------------------------------------------------
    # Setup switcher selector if needed
    # ------------------------------------------------------------------
    switcher_selector = None
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

        extra_features = config["extra_features"]
        scalar_dim = 1 + 3 + len(extra_features)

        feature_builder = GroupFeatureBuilder(
            embed_dim=embed_dim,
            extra_feature_names=extra_features,
            extra_aggregations=config["extra_aggregations"],
        )

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

        switcher_selector = SwitcherGroupSelector(
            switcher=switcher,
            feature_builder=feature_builder,
            policy=policy,
            groups=groups,
            device=device,
            top_k=config.get("top_k_selection", 1),
        )
        logger.info(
            f"Switcher selector will randomly choose from top "
            f"{config.get('top_k_selection', 1)} groups"
        )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
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
        seed=config["seed"],
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    summary = stats.get_summary()

    logger.info("\n" + "=" * 70)
    logger.info(f"SWITCHER TEST RESULTS (v2) - Mode: {config['selection_mode'].upper()}")
    logger.info("=" * 70)

    logger.info("\n--- Episode Statistics ---")
    logger.info(f"Episodes: {summary['num_episodes']}")
    logger.info(f"Average episode reward: {summary['avg_episode_reward']:.2f}")
    logger.info(f"Average steps per episode: {summary['avg_steps']:.1f}")

    logger.info("\n--- Episode-Level Outcomes ---")
    logger.info(f"  Success (all robots reached goal, 0 collisions): "
                f"{summary['success_count']}/{summary['num_episodes']}")
    logger.info(f"  Collision (episode ended by collision/OOB):       "
                f"{summary['collision_count']}/{summary['num_episodes']}")
    logger.info(f"  Timeout  (max steps without all goals reached):   "
                f"{summary['timeout_count']}/{summary['num_episodes']}")

    logger.info(f"\n  SUCCESS  RATE: {summary['success_rate']:.2%}")
    logger.info(f"  COLLISION RATE: {summary['collision_rate']:.2%}")
    logger.info(f"  TIMEOUT  RATE: {summary['timeout_rate']:.2%}")

    logger.info(f"\n  Avg reward (success episodes):   {summary['avg_success_reward']:.2f}")
    logger.info(f"  Avg reward (collision episodes):  {summary['avg_collision_reward']:.2f}")
    logger.info(f"  Avg reward (timeout episodes):    {summary['avg_timeout_reward']:.2f}")
    logger.info(f"  Avg steps  (success episodes):    {summary['avg_success_steps']:.1f}")

    logger.info("\n--- Path Length Statistics (Success Episodes Only) ---")
    if summary["num_success_episodes_for_path"] > 0:
        logger.info(f"  Based on {summary['num_success_episodes_for_path']} successful episodes")
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
        logger.info("  No successful episodes — path length unavailable.")

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

    logger.info("\n" + "=" * 70)

    return stats


if __name__ == "__main__":
    main()
