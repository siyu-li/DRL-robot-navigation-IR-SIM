"""
RL (PPO) group switcher components.

- ``RLFeatureBuilder``: produces per-group features for actor + state features
  for critic.
- ``SwitcherActorCritic`` / ``SwitcherPPO``: PPO training framework with
  categorical action space over candidate groups.
- ``SwitcherEnv``: Gym-like environment wrapper that runs the decentralized TD3
  policy for ``selection_interval`` sim steps and returns swarm-level rewards.
"""

from robot_nav.models.MARL.switcher.rl.rl_feature_builder import (
    RLFeatureBuilder,
)
from robot_nav.models.MARL.switcher.rl.rl_attn_feature_builder import (
    RLAttnFeatureBuilder,
)
from robot_nav.models.MARL.switcher.rl.switcher_ppo import (
    SwitcherActorCritic,
    SwitcherPPO,
    SwitcherRolloutBuffer,
)
from robot_nav.models.MARL.switcher.rl.switcher_env import SwitcherEnv

__all__ = [
    "RLFeatureBuilder",
    "RLAttnFeatureBuilder",
    "SwitcherActorCritic",
    "SwitcherPPO",
    "SwitcherRolloutBuffer",
    "SwitcherEnv",
]
