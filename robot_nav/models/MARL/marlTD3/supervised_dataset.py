"""
Supervised Training Dataset for Coupled Action Policy.

This module provides utilities to:
1. Generate v_shared labels from decentralized rollout data using aggregation functions
2. Create PyTorch datasets for supervised training of the v_head
"""

from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Tuple
from collections import deque
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm


# Supported aggregation modes for generating v_shared labels
V_LABEL_MODES = Literal["p10", "p20", "p30", "mean", "min"]


def compute_v_label(
    v_values: np.ndarray,
    mode: V_LABEL_MODES = "p20"
) -> float:
    """
    Compute aggregated v_shared label from per-robot linear velocities.
    
    Args:
        v_values (np.ndarray): Per-robot linear velocities, shape (N,).
        mode (str): Aggregation mode:
            - "p10": 10th percentile
            - "p20": 20th percentile (default)
            - "p30": 30th percentile
            - "mean": arithmetic mean
            - "min": minimum value
            
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
    else:
        raise ValueError(f"Unknown v_label_mode: {mode}")


class SupervisedVDataset(Dataset):
    """
    PyTorch Dataset for supervised training of v_head.
    
    Each sample contains:
    - state: Robot states for all robots, shape (N, state_dim)
    - v_label: Aggregated v_shared label, scalar
    
    Args:
        states (np.ndarray): Array of states, shape (num_samples, N, state_dim).
        v_labels (np.ndarray): Array of v_shared labels, shape (num_samples,).
    """
    
    def __init__(self, states: np.ndarray, v_labels: np.ndarray):
        assert len(states) == len(v_labels), \
            f"States and labels must have same length: {len(states)} vs {len(v_labels)}"
        
        self.states = torch.FloatTensor(states)
        self.v_labels = torch.FloatTensor(v_labels).unsqueeze(-1)  # (N,) -> (N, 1)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.v_labels[idx]


class SupervisedDatasetGenerator:
    """
    Generates supervised training datasets from decentralized rollout data.
    
    Reads YAML files containing rollout data and creates (state, v_label) pairs
    where v_label is computed by aggregating teacher linear velocities.
    
    Args:
        file_paths (List[str]): Paths to YAML rollout data files.
        num_robots (int): Number of robots in the data.
        state_dim (int): Per-robot state dimension.
        v_label_mode (str): Aggregation mode for v_label computation.
        v_min (float): Minimum valid linear velocity (for clipping labels).
        v_max (float): Maximum valid linear velocity (for clipping labels).
    """
    
    def __init__(
        self,
        file_paths: List[str],
        num_robots: int = 5,
        state_dim: int = 11,
        v_label_mode: V_LABEL_MODES = "p20",
        v_min: float = 0.0,
        v_max: float = 0.5
    ):
        self.file_paths = file_paths
        self.num_robots = num_robots
        self.state_dim = state_dim
        self.v_label_mode = v_label_mode
        self.v_min = v_min
        self.v_max = v_max
    
    def _prepare_state_vector(
        self,
        pose: List[float],
        distance: float,
        cos_val: float,
        sin_val: float,
        action: List[float],
        goal_pos: List[float]
    ) -> List[float]:
        """
        Prepare state vector for a single robot.
        
        Args:
            pose: [x, y, theta]
            distance: Distance to goal
            cos_val: cos(heading error)
            sin_val: sin(heading error)
            action: [lin_vel, ang_vel] in a_in format (lin_vel in [0, 0.5])
            goal_pos: [gx, gy]
            
        Returns:
            State vector of length state_dim.
        """
        px, py, theta = pose
        gx, gy = goal_pos
        
        heading_cos = np.cos(theta)
        heading_sin = np.sin(theta)
        
        # Convert from a_in format to state format
        lin_vel = action[0] * 2  # a_in lin_vel is in [0, 0.5], multiply by 2
        ang_vel = (action[1] + 1) / 2  # a_in ang_vel is in [-1, 1], shift to [0, 1]
        
        state = [
            px, py,
            heading_cos, heading_sin,
            distance / 17,  # Normalize distance
            cos_val, sin_val,
            lin_vel, ang_vel,
            gx, gy,
        ]
        
        return state
    
    def _extract_teacher_v(self, actions: List[List[float]]) -> np.ndarray:
        """
        Extract teacher linear velocities from raw actions.
        
        Args:
            actions: List of [lin_vel, ang_vel] per robot in raw format [-1, 1].
            
        Returns:
            np.ndarray: Per-robot linear velocities in [0, 0.5] range.
        """
        # Raw actions are in [-1, 1], convert to a_in format [0, 0.5] for linear velocity
        v_values = np.array([(a[0] + 1) / 4 for a in actions])
        return v_values
    
    def generate_from_yaml(
        self,
        skip_collisions: bool = True,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset from YAML rollout files.
        
        Args:
            skip_collisions: If True, skip timesteps with any collision.
            max_samples: Maximum number of samples to generate (None = no limit).
            
        Returns:
            Tuple of (states, v_labels):
                - states: shape (num_samples, N, state_dim)
                - v_labels: shape (num_samples,)
        """
        states_list = []
        v_labels_list = []
        
        for file_path in self.file_paths:
            print(f"Loading rollout data from: {file_path}")
            
            with open(file_path, "r") as f:
                samples = yaml.full_load(f)
            
            if samples is None:
                print(f"  Warning: Empty file {file_path}")
                continue
            
            for sample_id, sample in tqdm(samples.items(), desc="Processing samples"):
                # Skip if collision occurred
                if skip_collisions and any(sample.get("collisions", [])):
                    continue
                
                poses = sample["poses"]
                distances = sample["distances"]
                cos_vals = sample["cos"]
                sin_vals = sample["sin"]
                actions = sample["actions"]  # Raw actions in [-1, 1]
                goal_positions = sample["goal_positions"]
                
                # Validate robot count
                if len(poses) != self.num_robots:
                    continue
                
                # Build per-robot states
                robot_states = []
                for i in range(self.num_robots):
                    # Convert raw action to a_in format for state preparation
                    action_ain = [(actions[i][0] + 1) / 4, actions[i][1]]
                    
                    state = self._prepare_state_vector(
                        pose=poses[i],
                        distance=distances[i],
                        cos_val=cos_vals[i],
                        sin_val=sin_vals[i],
                        action=action_ain,
                        goal_pos=goal_positions[i]
                    )
                    robot_states.append(state)
                
                # Compute v_label from teacher velocities
                teacher_v = self._extract_teacher_v(actions)
                v_label = compute_v_label(teacher_v, self.v_label_mode)
                
                # Clip to valid range
                v_label = np.clip(v_label, self.v_min, self.v_max)
                
                states_list.append(robot_states)
                v_labels_list.append(v_label)
                
                if max_samples and len(states_list) >= max_samples:
                    break
            
            if max_samples and len(states_list) >= max_samples:
                break
        
        states = np.array(states_list, dtype=np.float32)
        v_labels = np.array(v_labels_list, dtype=np.float32)
        
        print(f"Generated {len(states)} samples")
        print(f"  States shape: {states.shape}")
        print(f"  V labels shape: {v_labels.shape}")
        print(f"  V label stats: min={v_labels.min():.4f}, max={v_labels.max():.4f}, mean={v_labels.mean():.4f}")
        
        return states, v_labels
    

def create_dataloader(
    states: np.ndarray,
    v_labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    train_split: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from arrays.
    
    Args:
        states: State array, shape (N, num_robots, state_dim).
        v_labels: Label array, shape (N,).
        batch_size: Batch size for DataLoaders.
        shuffle: Whether to shuffle training data.
        train_split: Fraction of data for training.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Split data
    n_samples = len(states)
    n_train = int(n_samples * train_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = SupervisedVDataset(states[train_indices], v_labels[train_indices])
    val_dataset = SupervisedVDataset(states[val_indices], v_labels[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
