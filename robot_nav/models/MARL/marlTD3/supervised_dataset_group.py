"""
Supervised Training Dataset for Group-based Coupled Action Policy.

This module extends supervised_dataset.py to support:
1. Group-based v_shared labels (pooling only from robots in a group)
2. Loading data from replay buffers (avoiding YAML serialization issues)
3. Obstacle-aware data handling (from marl_train_obstacle_6robots)
4. Flexible group definitions (predefined or all combinations)

Key Difference from Original:
- Original: v_label = aggregate(v_all_robots) → G = mean(H_all)
- Group:    v_label = aggregate(v_group_robots) → G = mean(H_group)
"""

from itertools import combinations
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Tuple, Union
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Supported aggregation modes for generating v_shared labels
V_LABEL_MODES = Literal["p10", "p20", "p30", "mean", "min", "max"]


def compute_v_label(
    v_values: np.ndarray,
    mode: V_LABEL_MODES = "mean"
) -> float:
    """
    Compute aggregated v_shared label from per-robot linear velocities.
    
    Args:
        v_values (np.ndarray): Per-robot linear velocities for the GROUP, shape (group_size,).
        mode (str): Aggregation mode:
            - "p10": 10th percentile
            - "p20": 20th percentile
            - "p30": 30th percentile
            - "mean": arithmetic mean (default)
            - "min": minimum value
            - "max": maximum value
            
    Returns:
        float: Aggregated v_shared label.
    """
    if mode == "p10":
        return float(np.percentile(v_values, 10))
    elif mode == "p20":
        return float(np.percentile(v_values, 20))
    elif mode == "p30":
        return float(np.percentile(v_values, 30))
    elif mode == "mean":
        return float(np.mean(v_values))
    elif mode == "min":
        return float(np.min(v_values))
    elif mode == "max":
        return float(np.max(v_values))
    else:
        raise ValueError(f"Unknown v_label_mode: {mode}")


def generate_all_groups(num_robots: int, group_sizes: List[int] = [2, 3]) -> List[List[int]]:
    """
    Generate all possible robot group combinations.
    
    Args:
        num_robots (int): Total number of robots.
        group_sizes (List[int]): List of group sizes to generate.
        
    Returns:
        List[List[int]]: List of robot index groups.
        
    Example:
        >>> generate_all_groups(6, [2, 3])
        [[0, 1], [0, 2], ..., [0, 1, 2], [0, 1, 3], ...]
    """
    all_groups = []
    robot_indices = list(range(num_robots))
    
    for size in group_sizes:
        if size <= num_robots:
            for combo in combinations(robot_indices, size):
                all_groups.append(list(combo))
    
    return all_groups


class SupervisedGroupVDataset(Dataset):
    """
    PyTorch Dataset for supervised training of group-based v_head.
    
    Each sample contains:
    - robot_states: Robot states for ALL robots, shape (N, state_dim)
    - obstacle_states: Obstacle states, shape (M, obstacle_state_dim) or None
    - group_indices: Indices of robots in the group, shape (group_size,)
    - v_label: Aggregated v_shared label for the GROUP, scalar
    
    Args:
        robot_states (np.ndarray): Array of robot states, shape (num_samples, N, state_dim).
        obstacle_states (np.ndarray or None): Array of obstacle states, shape (num_samples, M, obs_dim).
        group_indices (np.ndarray): Array of group indices, shape (num_samples, group_size).
        v_labels (np.ndarray): Array of v_shared labels, shape (num_samples,).
    """
    
    def __init__(
        self, 
        robot_states: np.ndarray, 
        obstacle_states: Optional[np.ndarray],
        group_indices: np.ndarray,
        v_labels: np.ndarray
    ):
        assert len(robot_states) == len(v_labels), \
            f"States and labels must have same length: {len(robot_states)} vs {len(v_labels)}"
        assert len(robot_states) == len(group_indices), \
            f"States and group_indices must have same length"
        
        self.robot_states = torch.FloatTensor(robot_states)
        self.obstacle_states = torch.FloatTensor(obstacle_states) if obstacle_states is not None else None
        self.group_indices = torch.LongTensor(group_indices)
        self.v_labels = torch.FloatTensor(v_labels).unsqueeze(-1)  # (N,) -> (N, 1)
    
    def __len__(self) -> int:
        return len(self.robot_states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        obs_state = self.obstacle_states[idx] if self.obstacle_states is not None else None
        return self.robot_states[idx], obs_state, self.group_indices[idx], self.v_labels[idx]


def collate_group_batch(batch):
    """
    Custom collate function for variable group sizes.
    
    Since group_indices may have different sizes across samples, we need custom collation.
    For simplicity, this version assumes uniform group size within a batch.
    """
    robot_states = torch.stack([item[0] for item in batch])
    
    # Handle obstacle states
    if batch[0][1] is not None:
        obstacle_states = torch.stack([item[1] for item in batch])
    else:
        obstacle_states = None
    
    group_indices = torch.stack([item[2] for item in batch])
    v_labels = torch.stack([item[3] for item in batch])
    
    return robot_states, obstacle_states, group_indices, v_labels


class SupervisedGroupDatasetGenerator:
    """
    Generates supervised training datasets for group-based coupled action policy.
    
    Supports loading from:
    1. Saved replay buffers (ReplayBufferObstacle pickles)
    2. Direct buffer object
    
    For each timestep, generates multiple samples - one per group.
    
    Args:
        num_robots (int): Number of robots in the data.
        state_dim (int): Per-robot state dimension.
        obstacle_state_dim (int): Per-obstacle state dimension.
        v_label_mode (str): Aggregation mode for v_label computation.
        v_min (float): Minimum valid linear velocity.
        v_max (float): Maximum valid linear velocity.
        groups (List[List[int]] or None): Predefined groups. If None, generates all combinations.
        group_sizes (List[int]): Group sizes to generate if groups is None.
    """
    
    def __init__(
        self,
        num_robots: int = 6,
        state_dim: int = 11,
        obstacle_state_dim: int = 4,
        v_label_mode: V_LABEL_MODES = "mean",
        v_min: float = 0.0,
        v_max: float = 0.5,
        groups: Optional[List[List[int]]] = None,
        group_sizes: List[int] = [2, 3],
    ):
        self.num_robots = num_robots
        self.state_dim = state_dim
        self.obstacle_state_dim = obstacle_state_dim
        self.v_label_mode = v_label_mode
        self.v_min = v_min
        self.v_max = v_max
        
        # Generate groups if not provided
        if groups is None:
            self.groups = generate_all_groups(num_robots, group_sizes)
        else:
            self.groups = groups
        
        print(f"Initialized with {len(self.groups)} groups:")
        for g in self.groups[:5]:
            print(f"  Group: {g}")
        if len(self.groups) > 5:
            print(f"  ... and {len(self.groups) - 5} more groups")
    
    def _extract_v_from_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Extract linear velocities from actions stored in replay buffer.
        
        The replay buffer stores actions in raw format [-1, 1].
        We convert to [0, 0.5] range for linear velocity.
        
        Args:
            actions (np.ndarray): Actions of shape (N_robots, 2) in [-1, 1] range.
            
        Returns:
            np.ndarray: Per-robot linear velocities in [0, 0.5] range.
        """
        # actions[:, 0] is linear velocity in [-1, 1]
        # Convert to [0, 0.5]: (v + 1) / 4
        v_values = (actions[:, 0] + 1) / 4
        return v_values
    
    def generate_from_buffer(
        self,
        buffer_or_path: Union[str, Path, "ReplayBufferObstacle"],
        skip_terminal: bool = True,
        max_samples_per_group: Optional[int] = None,
        sample_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate dataset from replay buffer.
        
        For each timestep in the buffer, generates one sample per group.
        This creates len(buffer) * len(groups) total samples.
        
        Args:
            buffer_or_path: Either a ReplayBufferObstacle object or path to pickle file.
            skip_terminal: If True, skip terminal states (collisions).
            max_samples_per_group: Max samples to use per group (None = no limit).
            sample_fraction: Fraction of buffer to use (for large buffers).
            
        Returns:
            Tuple of (robot_states, obstacle_states, group_indices, v_labels):
                - robot_states: shape (num_samples, N_robots, state_dim)
                - obstacle_states: shape (num_samples, N_obstacles, obstacle_state_dim)
                - group_indices: shape (num_samples, max_group_size) - padded with -1
                - v_labels: shape (num_samples,)
        """
        # Load buffer if path provided
        if isinstance(buffer_or_path, (str, Path)):
            print(f"Loading replay buffer from: {buffer_or_path}")
            with open(buffer_or_path, "rb") as f:
                buffer = pickle.load(f)
        else:
            buffer = buffer_or_path
        
        print(f"Buffer size: {buffer.size()}")
        
        # Extract all data from buffer
        (
            all_robot_states,
            all_obstacle_states,
            all_actions,
            all_rewards,
            all_dones,
            all_next_robot_states,
            all_next_obstacle_states,
        ) = buffer.return_buffer()
        
        print(f"Loaded data shapes:")
        print(f"  Robot states: {all_robot_states.shape}")
        print(f"  Obstacle states: {all_obstacle_states.shape}")
        print(f"  Actions: {all_actions.shape}")
        
        # Sample subset if requested
        n_total = len(all_robot_states)
        if sample_fraction < 1.0:
            n_sample = int(n_total * sample_fraction)
            sample_indices = np.random.choice(n_total, n_sample, replace=False)
            all_robot_states = all_robot_states[sample_indices]
            all_obstacle_states = all_obstacle_states[sample_indices]
            all_actions = all_actions[sample_indices]
            all_dones = all_dones[sample_indices]
            print(f"Sampled {n_sample} / {n_total} timesteps")
        
        # Find max group size for padding
        max_group_size = max(len(g) for g in self.groups)
        
        # Generate samples
        robot_states_list = []
        obstacle_states_list = []
        group_indices_list = []
        v_labels_list = []
        
        samples_per_group = {tuple(g): 0 for g in self.groups}
        
        for t_idx in tqdm(range(len(all_robot_states)), desc="Processing timesteps"):
            # Skip terminal states if requested
            if skip_terminal and np.any(all_dones[t_idx]):
                continue
            
            robot_state = all_robot_states[t_idx]  # (N_robots, state_dim)
            obstacle_state = all_obstacle_states[t_idx]  # (N_obs, obstacle_state_dim)
            action = all_actions[t_idx]  # (N_robots, action_dim)
            
            # Extract per-robot velocities
            v_all = self._extract_v_from_actions(action)
            
            # Generate one sample per group
            for group in self.groups:
                group_key = tuple(group)
                
                # Check max samples limit
                if max_samples_per_group and samples_per_group[group_key] >= max_samples_per_group:
                    continue
                
                # Extract velocities for this group only
                v_group = v_all[group]
                
                # Compute group-specific label
                v_label = compute_v_label(v_group, self.v_label_mode)
                v_label = np.clip(v_label, self.v_min, self.v_max)
                
                # Pad group indices to max_group_size (use -1 for padding)
                padded_group = list(group) + [-1] * (max_group_size - len(group))
                
                robot_states_list.append(robot_state)
                obstacle_states_list.append(obstacle_state)
                group_indices_list.append(padded_group)
                v_labels_list.append(v_label)
                
                samples_per_group[group_key] += 1
        
        # Convert to arrays
        robot_states = np.array(robot_states_list, dtype=np.float32)
        obstacle_states = np.array(obstacle_states_list, dtype=np.float32)
        group_indices = np.array(group_indices_list, dtype=np.int64)
        v_labels = np.array(v_labels_list, dtype=np.float32)
        
        print(f"\nGenerated {len(robot_states)} samples total")
        print(f"  Robot states shape: {robot_states.shape}")
        print(f"  Obstacle states shape: {obstacle_states.shape}")
        print(f"  Group indices shape: {group_indices.shape}")
        print(f"  V labels shape: {v_labels.shape}")
        print(f"  V label stats: min={v_labels.min():.4f}, max={v_labels.max():.4f}, mean={v_labels.mean():.4f}")
        
        print(f"\nSamples per group:")
        for g, count in list(samples_per_group.items())[:10]:
            print(f"  Group {list(g)}: {count}")
        
        return robot_states, obstacle_states, group_indices, v_labels
    
    def generate_uniform_group_samples(
        self,
        buffer_or_path: Union[str, Path, "ReplayBufferObstacle"],
        samples_per_group: int = 5000,
        skip_terminal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate dataset with uniform samples per group.
        
        Ensures each group has the same number of samples by randomly
        sampling from the buffer for each group.
        
        Args:
            buffer_or_path: Either a ReplayBufferObstacle object or path to pickle file.
            samples_per_group: Number of samples per group.
            skip_terminal: If True, skip terminal states.
            
        Returns:
            Same as generate_from_buffer.
        """
        # Load buffer
        if isinstance(buffer_or_path, (str, Path)):
            print(f"Loading replay buffer from: {buffer_or_path}")
            with open(buffer_or_path, "rb") as f:
                buffer = pickle.load(f)
        else:
            buffer = buffer_or_path
        
        # Extract all data
        (
            all_robot_states,
            all_obstacle_states,
            all_actions,
            all_rewards,
            all_dones,
            _,
            _,
        ) = buffer.return_buffer()
        
        # Get valid indices (non-terminal)
        if skip_terminal:
            # all_dones has shape (N, N_robots) - skip if any robot is done
            valid_mask = ~np.any(all_dones.reshape(len(all_dones), -1), axis=1)
            valid_indices = np.where(valid_mask)[0]
        else:
            valid_indices = np.arange(len(all_robot_states))
        
        print(f"Valid timesteps: {len(valid_indices)} / {len(all_robot_states)}")
        
        max_group_size = max(len(g) for g in self.groups)
        
        robot_states_list = []
        obstacle_states_list = []
        group_indices_list = []
        v_labels_list = []
        
        for group in tqdm(self.groups, desc="Processing groups"):
            # Sample indices for this group
            if samples_per_group <= len(valid_indices):
                sampled = np.random.choice(valid_indices, samples_per_group, replace=False)
            else:
                sampled = np.random.choice(valid_indices, samples_per_group, replace=True)
            
            for t_idx in sampled:
                robot_state = all_robot_states[t_idx]
                obstacle_state = all_obstacle_states[t_idx]
                action = all_actions[t_idx]
                
                v_all = self._extract_v_from_actions(action)
                v_group = v_all[group]
                v_label = compute_v_label(v_group, self.v_label_mode)
                v_label = np.clip(v_label, self.v_min, self.v_max)
                
                padded_group = list(group) + [-1] * (max_group_size - len(group))
                
                robot_states_list.append(robot_state)
                obstacle_states_list.append(obstacle_state)
                group_indices_list.append(padded_group)
                v_labels_list.append(v_label)
        
        robot_states = np.array(robot_states_list, dtype=np.float32)
        obstacle_states = np.array(obstacle_states_list, dtype=np.float32)
        group_indices = np.array(group_indices_list, dtype=np.int64)
        v_labels = np.array(v_labels_list, dtype=np.float32)
        
        print(f"\nGenerated {len(robot_states)} samples ({samples_per_group} per group × {len(self.groups)} groups)")
        print(f"  V label stats: min={v_labels.min():.4f}, max={v_labels.max():.4f}, mean={v_labels.mean():.4f}")
        
        return robot_states, obstacle_states, group_indices, v_labels


def create_group_dataloader(
    robot_states: np.ndarray,
    obstacle_states: np.ndarray,
    group_indices: np.ndarray,
    v_labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    train_split: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for group-based supervised learning.
    
    Args:
        robot_states: Robot state array, shape (N, num_robots, state_dim).
        obstacle_states: Obstacle state array, shape (N, num_obstacles, obs_dim).
        group_indices: Group index array, shape (N, max_group_size).
        v_labels: Label array, shape (N,).
        batch_size: Batch size for DataLoaders.
        shuffle: Whether to shuffle training data.
        train_split: Fraction of data for training.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    n_samples = len(robot_states)
    n_train = int(n_samples * train_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = SupervisedGroupVDataset(
        robot_states[train_indices],
        obstacle_states[train_indices] if obstacle_states is not None else None,
        group_indices[train_indices],
        v_labels[train_indices]
    )
    val_dataset = SupervisedGroupVDataset(
        robot_states[val_indices],
        obstacle_states[val_indices] if obstacle_states is not None else None,
        group_indices[val_indices],
        v_labels[val_indices]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_group_batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_group_batch
    )
    
    return train_loader, val_loader


# ============================================================================
# Utility functions for data collection during training
# ============================================================================

class OnlineGroupDataCollector:
    """
    Collects data online during environment rollouts for group supervised learning.
    
    Instead of saving to YAML (which fails for large datasets), this collector
    stores data in memory and can save to pickle files in chunks.
    
    Usage:
        collector = OnlineGroupDataCollector(...)
        
        # During training loop:
        collector.add(robot_state, obstacle_state, action, done)
        
        # Periodically save:
        collector.save_chunk("data_chunk_001.pkl")
    """
    
    def __init__(
        self,
        num_robots: int = 6,
        max_size: int = 100000,
        save_directory: Path = Path("robot_nav/assets/group_data"),
    ):
        self.num_robots = num_robots
        self.max_size = max_size
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        self.robot_states = []
        self.obstacle_states = []
        self.actions = []
        self.dones = []
        
        self.chunk_count = 0
    
    def add(
        self,
        robot_state: np.ndarray,
        obstacle_state: np.ndarray,
        action: np.ndarray,
        done: List[bool],
    ):
        """Add a single timestep of data."""
        self.robot_states.append(np.array(robot_state))
        self.obstacle_states.append(np.array(obstacle_state))
        self.actions.append(np.array(action))
        self.dones.append(np.array(done))
        
        # Auto-save if max size reached
        if len(self.robot_states) >= self.max_size:
            self.save_chunk()
    
    def size(self) -> int:
        return len(self.robot_states)
    
    def save_chunk(self, filename: Optional[str] = None):
        """Save current data to a pickle file and clear memory."""
        if len(self.robot_states) == 0:
            return
        
        if filename is None:
            filename = f"group_data_chunk_{self.chunk_count:04d}.pkl"
        
        data = {
            "robot_states": np.array(self.robot_states),
            "obstacle_states": np.array(self.obstacle_states),
            "actions": np.array(self.actions),
            "dones": np.array(self.dones),
        }
        
        filepath = self.save_directory / filename
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.robot_states)} samples to {filepath}")
        
        # Clear memory
        self.robot_states = []
        self.obstacle_states = []
        self.actions = []
        self.dones = []
        self.chunk_count += 1
    
    def load_all_chunks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and concatenate all saved chunks."""
        chunk_files = sorted(self.save_directory.glob("group_data_chunk_*.pkl"))
        
        all_robot_states = []
        all_obstacle_states = []
        all_actions = []
        all_dones = []
        
        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as f:
                data = pickle.load(f)
            all_robot_states.append(data["robot_states"])
            all_obstacle_states.append(data["obstacle_states"])
            all_actions.append(data["actions"])
            all_dones.append(data["dones"])
        
        return (
            np.concatenate(all_robot_states),
            np.concatenate(all_obstacle_states),
            np.concatenate(all_actions),
            np.concatenate(all_dones),
        )
