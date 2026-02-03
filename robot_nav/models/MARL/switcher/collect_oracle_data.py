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
    "n_samples": 1000,              # Number of samples to collect
    "n_robots": 6,                  # Number of robots
    "n_obstacles": 4,               # Number of obstacles
    "embed_dim": 256,               # Embedding dimension from GAT backbone
    "seed": 42,                     # Random seed for reproducibility
    
    # Oracle evaluation settings
    "oracle_horizon": 10,           # Number of steps to simulate forward for each group
    "n_rollouts_per_group": 3,      # Number of rollouts to average for each group score
    
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
    "max_steps_per_episode": 500,   # Reset episode after this many steps
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
        averaging the linear velocities of all robots in the group.
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
    ) -> Tuple[float, bool]:
        """
        Evaluate a group by simulating forward H steps (single rollout).
        
        Args:
            group: Robot indices in the group.
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after rollout.
            
        Returns:
            Tuple of (cumulative_reward, had_collision).
        """
        cumulative_reward = 0.0
        had_collision = False
        
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
            
            # Accumulate reward for robots in the group
            group_reward = sum(reward[i] for i in group)
            cumulative_reward += group_reward
            
            # Check for collision - end rollout early if collision
            if any(curr_collision[i] for i in group):
                had_collision = True
                break
            
            # Check for out of bounds
            if outside_of_bounds(curr_poses, self.sim):
                had_collision = True
                break
        
        # Restore simulation to original state
        snapshot.restore_to_sim(self.sim)
        
        return cumulative_reward, had_collision
    
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
    ) -> float:
        """
        Evaluate a group by averaging over n_rollouts.
        
        Args:
            group: List of robot indices in the group
            poses, distance, cos, sin, collision, action, goal_positions, obstacle_states:
                Current environment state.
            snapshot: Simulation snapshot to restore after each rollout.
            
        Returns:
            score: Average cumulative reward across rollouts (higher = better)
        """
        total_reward = 0.0
        
        for _ in range(self.n_rollouts_per_group):
            reward, _ = self._evaluate_group_once(
                group, poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states, snapshot
            )
            total_reward += reward
        
        avg_reward = total_reward / self.n_rollouts_per_group
        return avg_reward
    
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
        
        # Evaluate each group with rollouts
        group_scores = []
        
        for group in self.groups:
            score = self._evaluate_group(
                group, poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states, snapshot
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
    ) -> Dict:
        """
        Collect a full dataset of oracle samples by running episodes.
        
        Args:
            n_samples: Number of samples to collect
            save_path: Path to save the dataset (optional)
            verbose: Print progress
            
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
        max_steps = CONFIG.get("max_steps_per_episode", 500)
        
        for i in pbar:
            # Collect sample at current state
            sample = self.collect_sample(
                poses, distance, cos, sin, collision, action,
                goal_positions, obstacle_states,
                scenario_id=i
            )
            samples.append(sample)
            
            # Take a random action to advance simulation to a new state
            # Use decentralized policy to get diverse states
            robot_state, _ = self.policy.prepare_state(
                poses, distance, cos, sin, collision, action, goal_positions
            )
            robot_obs = np.array(robot_state)
            
            action_out, _ = self.policy.get_action(
                robot_obs, obstacle_states, add_noise=True  # Add noise for diversity
            )
            
            # Scale actions
            scaled_action = []
            for j in range(self.num_robots):
                scaled_lin_vel = (action_out[j][0] + 1) / 4
                scaled_action.append([scaled_lin_vel, action_out[j][1]])
            
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
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states
                ) = self.sim.reset()
                step_in_episode = 0
        
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
    
    print("=" * 70)
    print("Oracle Data Collection for Group Switcher")
    print("=" * 70)
    print(f"Output path: {config['output_path']}")
    print(f"Number of samples: {config['n_samples']}")
    print(f"Number of robots: {config['n_robots']}")
    print(f"Embedding dimension: {config['embed_dim']}")
    print(f"Oracle horizon: {config['oracle_horizon']} steps")
    print(f"Rollouts per group: {config['n_rollouts_per_group']}")
    print("=" * 70 + "\n")
    

    # Real data collection using simulation
    print("Collecting REAL oracle data via simulation rollouts...")
    
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
