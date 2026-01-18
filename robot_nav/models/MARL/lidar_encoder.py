"""
LiDAR Encoder Module for MARL Navigation.

This module provides multiple options for encoding LiDAR scan data:
1. SectorEncoder: Divide scan into sectors, take min per sector (simplest, fastest)
2. LiDAREncoderMLP: MLP-based encoding (simple, good baseline)
3. LiDAREncoderCNN: 1D CNN-based encoding (captures spatial patterns)

Architecture Decision:
- LiDAR embeddings are fused AFTER the attention network (late fusion)
- Attention operates only on robot-to-robot relationships (edge features)
- LiDAR provides obstacle awareness directly to the policy head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal


# =============================================================================
# Option 1: Sector-Based Encoder (Simplest, Fastest, Most Interpretable)
# =============================================================================

class LiDARSectorEncoder(nn.Module):
    """
    Sector-based LiDAR encoder that divides the scan into sectors and extracts
    statistics (min, mean, or both) from each sector.

    This is the simplest and most interpretable approach:
    - Divide 180° scan into N sectors (e.g., 12 sectors = 15° each)
    - For each sector, compute min range (closest obstacle)
    - Optionally also compute mean range
    - Output is a small, interpretable vector

    Args:
        num_beams (int): Number of LiDAR beams (e.g., 180).
        num_sectors (int): Number of sectors to divide the scan into.
        output_dim (int): Output dimension. If > num_sectors, adds learned projection.
        aggregation (str): "min", "mean", or "both" for sector aggregation.
        learnable (bool): If True, adds a learnable linear layer after sector features.

    Example:
        - 180 beams, 12 sectors → each sector covers 15 beams (15°)
        - aggregation="min" → 12-dim output (closest obstacle per sector)
        - aggregation="both" → 24-dim output (min and mean per sector)
    """

    def __init__(
        self,
        num_beams: int = 180,
        num_sectors: int = 12,
        output_dim: Optional[int] = None,
        aggregation: Literal["min", "mean", "both"] = "min",
        learnable: bool = False,
    ):
        super().__init__()
        self.num_beams = num_beams
        self.num_sectors = num_sectors
        self.aggregation = aggregation
        self.learnable = learnable

        # Calculate beams per sector
        self.beams_per_sector = num_beams // num_sectors
        # Handle remainder beams (assign to last sector)
        self.remainder = num_beams % num_sectors

        # Determine raw feature size
        if aggregation == "both":
            self.raw_dim = num_sectors * 2  # min + mean per sector
        else:
            self.raw_dim = num_sectors

        # Output dimension
        self.output_dim = output_dim if output_dim is not None else self.raw_dim

        # Optional learnable projection
        if learnable or self.output_dim != self.raw_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.raw_dim, self.output_dim),
                nn.LeakyReLU(),
            )
            nn.init.kaiming_uniform_(self.projection[0].weight, nonlinearity="leaky_relu")
        else:
            self.projection = None

    def forward(self, lidar_scan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: divide scan into sectors and aggregate.

        Args:
            lidar_scan (Tensor): Shape (batch_size, num_beams) or (num_beams,).
                                 Values should be normalized to [0, 1].

        Returns:
            Tensor: Sector features of shape (batch_size, output_dim).
        """
        # Handle single sample
        if lidar_scan.dim() == 1:
            lidar_scan = lidar_scan.unsqueeze(0)

        batch_size = lidar_scan.shape[0]
        device = lidar_scan.device

        # Compute sector features
        sector_features = []

        start_idx = 0
        for i in range(self.num_sectors):
            # Last sector gets remainder beams
            if i == self.num_sectors - 1:
                end_idx = self.num_beams
            else:
                end_idx = start_idx + self.beams_per_sector

            sector_data = lidar_scan[:, start_idx:end_idx]  # (B, beams_in_sector)

            if self.aggregation == "min":
                # Minimum range in sector (closest obstacle)
                sector_val = sector_data.min(dim=1, keepdim=True)[0]
                sector_features.append(sector_val)
            elif self.aggregation == "mean":
                # Mean range in sector
                sector_val = sector_data.mean(dim=1, keepdim=True)
                sector_features.append(sector_val)
            else:  # "both"
                sector_min = sector_data.min(dim=1, keepdim=True)[0]
                sector_mean = sector_data.mean(dim=1, keepdim=True)
                sector_features.append(sector_min)
                sector_features.append(sector_mean)

            start_idx = end_idx

        # Concatenate all sector features
        features = torch.cat(sector_features, dim=1)  # (B, raw_dim)

        # Apply optional projection
        if self.projection is not None:
            features = self.projection(features)

        return features


# =============================================================================
# Option 2: MLP-Based Encoder (Simple, Good Baseline)
# =============================================================================

class LiDAREncoderMLP(nn.Module):
    """
    MLP-based encoder for LiDAR scan data.

    Simple feed-forward network that learns to compress LiDAR readings.
    Good baseline that's easy to train.

    Args:
        num_beams (int): Number of LiDAR beams (input size). Default: 180.
        output_dim (int): Output embedding dimension. Default: 32.
        hidden_dim (int): Hidden layer dimension. Default: 128.
        num_layers (int): Number of hidden layers. Default: 2.
        dropout (float): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        num_beams: int = 180,
        output_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_beams = num_beams
        self.output_dim = output_dim

        layers = []
        in_dim = num_beams

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.LeakyReLU())

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")

    def forward(self, lidar_scan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LiDAR MLP encoder.

        Args:
            lidar_scan (Tensor): Shape (batch_size, num_beams) or (num_beams,).

        Returns:
            Tensor: Embedding of shape (batch_size, output_dim).
        """
        if lidar_scan.dim() == 1:
            lidar_scan = lidar_scan.unsqueeze(0)
        return self.mlp(lidar_scan)


# =============================================================================
# Option 3: CNN-Based Encoder (Captures Spatial Patterns)
# =============================================================================

class LiDAREncoderCNN(nn.Module):
    """
    1D CNN encoder for LiDAR scan data.

    Convolutional approach that can capture spatial patterns in the scan
    (e.g., walls, corners, openings). Good when obstacle shapes matter.

    Args:
        num_beams (int): Number of LiDAR beams (input size). Default: 180.
        output_dim (int): Output embedding dimension. Default: 32.
        channels (list): List of channel sizes for conv layers. Default: [16, 32, 32].
    """

    def __init__(
        self,
        num_beams: int = 180,
        output_dim: int = 32,
        channels: Optional[list] = None,
    ):
        super().__init__()
        self.num_beams = num_beams
        self.output_dim = output_dim

        if channels is None:
            channels = [16, 32, 32]

        # Build convolutional layers
        conv_layers = []
        in_channels = 1

        for out_channels in channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            conv_layers.append(nn.LeakyReLU())
            in_channels = out_channels

        self.convs = nn.Sequential(*conv_layers)

        # Calculate output size after convolutions
        conv_out_size = self._get_conv_output_size(num_beams, len(channels))

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            nn.LeakyReLU(),
        )
        nn.init.kaiming_uniform_(self.fc[0].weight, nonlinearity="leaky_relu")

    def _get_conv_output_size(self, input_size: int, num_layers: int) -> int:
        """Calculate output size after conv layers (each halves the size)."""
        size = input_size
        for _ in range(num_layers):
            size = (size + 2 * 2 - 5) // 2 + 1  # Conv with k=5, s=2, p=2
        # Multiply by last channel count (assume 32)
        return size * 32

    def forward(self, lidar_scan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LiDAR CNN encoder.

        Args:
            lidar_scan (Tensor): Shape (batch_size, num_beams) or (num_beams,).

        Returns:
            Tensor: Embedding of shape (batch_size, output_dim).
        """
        if lidar_scan.dim() == 1:
            lidar_scan = lidar_scan.unsqueeze(0)

        # Add channel dimension: (B, num_beams) -> (B, 1, num_beams)
        x = lidar_scan.unsqueeze(1)

        # Convolutional layers
        x = self.convs(x)

        # Flatten and project
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


# =============================================================================
# Factory Function to Create Encoder
# =============================================================================

def create_lidar_encoder(
    encoder_type: str = "sector",
    num_beams: int = 180,
    output_dim: int = 12,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a LiDAR encoder.

    Args:
        encoder_type (str): One of "sector", "mlp", "cnn".
        num_beams (int): Number of LiDAR beams.
        output_dim (int): Output embedding dimension.
        **kwargs: Additional arguments passed to the encoder.

    Returns:
        nn.Module: LiDAR encoder module.

    Recommendations:
        - "sector": Simplest, fastest, most interpretable. Use 12 sectors for 180° scan.
                    Output is directly interpretable (min distance per direction).
        - "mlp": Good baseline, easy to train. Use for quick experiments.
        - "cnn": Best for capturing spatial patterns (walls, corners).
                 More parameters, may need more data.
    """
    if encoder_type == "sector":
        num_sectors = kwargs.get("num_sectors", 12)
        aggregation = kwargs.get("aggregation", "min")
        learnable = kwargs.get("learnable", False)
        return LiDARSectorEncoder(
            num_beams=num_beams,
            num_sectors=num_sectors,
            output_dim=output_dim,
            aggregation=aggregation,
            learnable=learnable,
        )
    elif encoder_type == "mlp":
        hidden_dim = kwargs.get("hidden_dim", 128)
        num_layers = kwargs.get("num_layers", 2)
        return LiDAREncoderMLP(
            num_beams=num_beams,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif encoder_type == "cnn":
        channels = kwargs.get("channels", [16, 32, 32])
        return LiDAREncoderCNN(
            num_beams=num_beams,
            output_dim=output_dim,
            channels=channels,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'sector', 'mlp', or 'cnn'.")


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_lidar_scan(
    scan: torch.Tensor, range_max: float = 7.0, clip_inf: bool = True
) -> torch.Tensor:
    """
    Normalize LiDAR scan data to [0, 1] range.

    Args:
        scan (Tensor): Raw LiDAR ranges.
        range_max (float): Maximum range value for normalization.
        clip_inf (bool): If True, replace inf values with range_max before normalizing.

    Returns:
        Tensor: Normalized scan in [0, 1] range.
    """
    if clip_inf:
        scan = torch.where(torch.isinf(scan), torch.full_like(scan, range_max), scan)
    return torch.clamp(scan / range_max, 0.0, 1.0)


def sector_min_numpy(scan: np.ndarray, num_sectors: int = 12) -> np.ndarray:
    """
    NumPy version of sector-based min aggregation for preprocessing.

    Useful for computing sector features before converting to tensor.

    Args:
        scan (np.ndarray): LiDAR scan of shape (num_beams,) or (batch, num_beams).
        num_sectors (int): Number of sectors.

    Returns:
        np.ndarray: Sector minimums of shape (num_sectors,) or (batch, num_sectors).
    """
    if scan.ndim == 1:
        scan = scan.reshape(1, -1)

    num_beams = scan.shape[1]
    beams_per_sector = num_beams // num_sectors
    remainder = num_beams % num_sectors

    sector_mins = []
    start_idx = 0

    for i in range(num_sectors):
        if i == num_sectors - 1:
            end_idx = num_beams
        else:
            end_idx = start_idx + beams_per_sector

        sector_data = scan[:, start_idx:end_idx]
        sector_mins.append(sector_data.min(axis=1, keepdims=True))
        start_idx = end_idx

    result = np.concatenate(sector_mins, axis=1)

    if result.shape[0] == 1:
        return result.squeeze(0)
    return result
