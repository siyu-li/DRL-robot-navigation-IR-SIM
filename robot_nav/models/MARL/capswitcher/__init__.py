"""
CAPSwitcher (Coarse-And-Precise Switcher) for coupled unicycle swarms.

Exports the complete CAPSwitcher implementation:

Policies (frozen backbone + coarse steering + readout)
  CoarseSteering       — least-squares pinv(A) group steering
  GATBackbone          — frozen TD3Obstacle actor wrapper (per-robot embeddings)
  DeepSetsHead         — permutation-invariant readout over per-robot embeddings
  
RL training (off-policy DQN)
  SwitcherEnv          — Gym-like environment (binary mode selection)
  DeepSetsQNet         — Deep Sets Q-network (per-robot → sum⊕max → Q-values)
  ReplayBuffer         — transition buffer storing per-robot embeddings
  SwitcherDQN          — Double-DQN trainer with TensorBoard logging

Usage:
    python -m robot_nav.marl_train_capswitcher
"""

from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.policies.deep_sets_head import DeepSetsHead
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv
from robot_nav.models.MARL.capswitcher.rl.switcher_dqn import (
    DeepSetsQNet,
    ReplayBuffer,
    SwitcherDQN,
)

__all__ = [
    "CoarseSteering",
    "GATBackbone",
    "DeepSetsHead",
    "SwitcherEnv",
    "DeepSetsQNet",
    "ReplayBuffer",
    "SwitcherDQN",
]
