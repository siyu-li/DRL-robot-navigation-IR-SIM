"""
CAPSwitcher (Coarse-And-Precise Switcher) for coupled unicycle swarms.

Exports the complete CAPSwitcher implementation:

Policies (frozen backbone + coarse steering + trained head)
  CoarseSteering       — least-squares pinv(A) group steering
  GATBackbone          — frozen TD3Obstacle actor wrapper
  SwitcherHead         — trainable MLP policy head (512→256→128→2)

RL training
  SwitcherEnv          — Gym-like environment (binary mode selection)
  SwitcherActorCritic  — shared-trunk actor + critic network
  SwitcherRolloutBuffer— transition buffer for PPO
  SwitcherPPO          — PPO update loop with TensorBoard logging

Usage:
    python -m robot_nav.marl_train_capswitcher
"""

from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.policies.cap_switcher import SwitcherHead

from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv
from robot_nav.models.MARL.capswitcher.rl.switcher_ppo import (
    SwitcherActorCritic,
    SwitcherRolloutBuffer,
    SwitcherPPO,
)

__all__ = [
    "CoarseSteering",
    "GATBackbone",
    "SwitcherHead",
    "SwitcherEnv",
    "SwitcherActorCritic",
    "SwitcherRolloutBuffer",
    "SwitcherPPO",
]
