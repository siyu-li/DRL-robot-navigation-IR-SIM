"""MARL Models Module - Multi-Agent RL Models for Navigation."""

from robot_nav.models.MARL.lidar_encoder import (
    LiDARSectorEncoder,
    LiDAREncoderMLP,
    LiDAREncoderCNN,
    create_lidar_encoder,
    normalize_lidar_scan,
)
from robot_nav.models.MARL.group_switch_planner import (
    GroupSwitchPlanner,
    GroupSwitchConfig,
    UrgencyCalculator,
    extract_lidar_sectors_from_state,
    generate_default_groups,
    get_positions_from_poses,
)

__all__ = [
    # LiDAR encoders
    "LiDARSectorEncoder",
    "LiDAREncoderMLP",
    "LiDAREncoderCNN",
    "create_lidar_encoder",
    "normalize_lidar_scan",
    # Group switching
    "GroupSwitchPlanner",
    "GroupSwitchConfig",
    "UrgencyCalculator",
    "extract_lidar_sectors_from_state",
    "generate_default_groups",
    "get_positions_from_poses",
]
