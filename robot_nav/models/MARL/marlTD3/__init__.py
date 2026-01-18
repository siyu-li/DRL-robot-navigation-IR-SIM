"""MARL TD3 models package."""

from robot_nav.models.MARL.marlTD3.marlTD3 import TD3
from robot_nav.models.MARL.marlTD3.marlTD3_lidar import TD3WithLiDAR
from robot_nav.models.MARL.marlTD3.coupled_action_policy import (
    CoupledActionPolicy,
    CoupledActionActor,
    SharedVelocityHead,
)

__all__ = [
    "TD3",
    "TD3WithLiDAR",
    "CoupledActionPolicy",
    "CoupledActionActor",
    "SharedVelocityHead",
]
