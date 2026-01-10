"""
Unit tests for Coupled Action Policy.

Tests the core functionality of the coupled action policy components.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_nav.models.MARL.marlTD3.coupled_action_policy import (
    SharedVelocityHead,
    CoupledActionActor,
    CoupledActionPolicy,
)
from robot_nav.models.MARL.marlTD3.supervised_dataset import (
    compute_v_label,
    SupervisedVDataset,
    SupervisedDatasetGenerator,
)


class TestSharedVelocityHead:
    """Tests for SharedVelocityHead module."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        head = SharedVelocityHead(input_dim=512, hidden_dim=128, v_min=0.0, v_max=0.5)
        x = torch.randn(32, 512)  # Batch of 32
        out = head(x)
        assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"
    
    def test_output_range(self):
        """Test that output is in [v_min, v_max] range."""
        v_min, v_max = 0.1, 0.4
        head = SharedVelocityHead(input_dim=512, hidden_dim=128, v_min=v_min, v_max=v_max)
        x = torch.randn(100, 512)
        out = head(x)
        assert out.min() >= v_min, f"Output below v_min: {out.min()}"
        assert out.max() <= v_max, f"Output above v_max: {out.max()}"


class TestCoupledActionActor:
    """Tests for CoupledActionActor module."""
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        actor = CoupledActionActor(
            embedding_dim=256,
            attention="igs",
            v_min=0.0,
            v_max=0.5,
            pooling="mean"
        )
        
        # Create mock observation: (batch, num_robots, obs_dim)
        batch_size, num_robots, obs_dim = 4, 5, 11
        obs = torch.randn(batch_size, num_robots, obs_dim)
        
        v_shared, omega, action, *_ = actor(obs)
        
        assert v_shared.shape == (batch_size, 1), f"v_shared shape: {v_shared.shape}"
        assert omega.shape == (batch_size * num_robots, 1), f"omega shape: {omega.shape}"
        assert action.shape == (batch_size * num_robots, 2), f"action shape: {action.shape}"
    
    def test_forward_2d_input(self):
        """Test forward pass with 2D input (no batch dimension)."""
        actor = CoupledActionActor(
            embedding_dim=256,
            attention="igs",
            v_min=0.0,
            v_max=0.5
        )
        
        num_robots, obs_dim = 5, 11
        obs = torch.randn(num_robots, obs_dim)  # 2D input
        
        v_shared, omega, action, *_ = actor(obs)
        
        assert v_shared.shape == (1, 1), f"v_shared shape: {v_shared.shape}"
        assert omega.shape == (num_robots, 1), f"omega shape: {omega.shape}"
    
    def test_v_shared_broadcast(self):
        """Test that v_shared is properly broadcast to all robots."""
        actor = CoupledActionActor(
            embedding_dim=256,
            attention="igs",
            v_min=0.0,
            v_max=0.5
        )
        
        batch_size, num_robots = 2, 5
        obs = torch.randn(batch_size, num_robots, 11)
        
        v_shared, omega, action, *_ = actor(obs)
        
        # Check that all robots in same batch have same v
        for b in range(batch_size):
            start_idx = b * num_robots
            end_idx = (b + 1) * num_robots
            robot_vs = action[start_idx:end_idx, 0]
            assert torch.allclose(robot_vs, robot_vs[0].expand_as(robot_vs)), \
                f"v_shared not uniform across robots in batch {b}"
    
    def test_get_v_shared_only(self):
        """Test get_v_shared_only method."""
        actor = CoupledActionActor(
            embedding_dim=256,
            attention="igs",
            v_min=0.0,
            v_max=0.5
        )
        
        obs = torch.randn(4, 5, 11)
        v_shared = actor.get_v_shared_only(obs)
        
        assert v_shared.shape == (4, 1)
    
    def test_pooling_methods(self):
        """Test different pooling methods."""
        for pooling in ["mean", "max"]:
            actor = CoupledActionActor(
                embedding_dim=256,
                attention="igs",
                v_min=0.0,
                v_max=0.5,
                pooling=pooling
            )
            obs = torch.randn(2, 5, 11)
            v_shared, *_ = actor(obs)
            assert v_shared.shape == (2, 1)


class TestVLabelComputation:
    """Tests for v_shared label computation."""
    
    def test_percentile_modes(self):
        """Test percentile computation modes."""
        v_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        p10 = compute_v_label(v_values, "p10")
        p20 = compute_v_label(v_values, "p20")
        p30 = compute_v_label(v_values, "p30")
        
        assert p10 < p20 < p30, "Percentiles should be ordered"
    
    def test_mean_mode(self):
        """Test mean aggregation."""
        v_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_v_label(v_values, "mean")
        expected = 0.3
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    def test_min_mode(self):
        """Test min aggregation."""
        v_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_v_label(v_values, "min")
        expected = 0.1
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"


class TestSupervisedVDataset:
    """Tests for PyTorch dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation and indexing."""
        states = np.random.randn(100, 5, 11).astype(np.float32)
        v_labels = np.random.uniform(0, 0.5, 100).astype(np.float32)
        
        dataset = SupervisedVDataset(states, v_labels)
        
        assert len(dataset) == 100
        
        state, label = dataset[0]
        assert state.shape == (5, 11)
        assert label.shape == (1,)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
