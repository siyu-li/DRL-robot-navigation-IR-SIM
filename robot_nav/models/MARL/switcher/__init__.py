"""
Group Switcher Module for Ranking-Based Group Selection.

This module provides two switcher variants:

**Supervised (oracle-based)**:
  - ``GroupFeatureBuilder``: constructs feature vectors for candidate groups.
  - ``GroupSwitcher``: two-tower fusion network for scoring/ranking groups.
  - Ranking losses for training.

**RL (PPO-based)**:
  - ``RLFeatureBuilder``: per-group + state features for actor-critic.
  - ``SwitcherActorCritic`` / ``SwitcherPPO``: PPO training framework.
  - ``SwitcherEnv``: Gym-like environment for group-selection RL.

Shared utilities (group generation, action coupling) live in
``robot_nav.models.MARL.groups``.

Usage examples:
    # Supervised training
    python -m robot_nav.scripts.train_switcher

    # RL (PPO) training
    python -m robot_nav.scripts.train_switcher_rl

    # Oracle data collection
    python -m robot_nav.scripts.collect_oracle_data_batch

    # Evaluation
    python -m robot_nav.scripts.test_switcher
"""

# ── Supervised switcher ──
from robot_nav.models.MARL.switcher.supervised import (
    GroupFeatureBuilder,
    _BASE_SCALAR_DIM,
    GroupSwitcher,
    pairwise_logistic_ranking_loss,
    hinge_ranking_loss,
    listwise_softmax_loss,
    build_pairs_from_scores,
    build_pairs_from_ranking,
    compute_ranking_accuracy,
    compute_top1_accuracy,
    RankingLossWithScheduledMargin,
)

# ── RL switcher ──
from robot_nav.models.MARL.switcher.rl import (
    RLFeatureBuilder,
    SwitcherActorCritic,
    SwitcherPPO,
    SwitcherRolloutBuffer,
    SwitcherEnv,
)

# ── Config ──
from robot_nav.models.MARL.switcher.config_loader import (
    SwitcherScalarConfig,
    load_switcher_config,
    config_from_dict,
)

# ── Groups (re-export for backward compatibility) ──
from robot_nav.models.MARL.groups import (
    generate_all_groups,
    generate_original_groups,
    generate_subgroups_recursive,
    filter_groups_by_size,
    print_group_statistics,
    actions_for_group,
    actions_for_group_from_raw,
)

__all__ = [
    # Supervised
    "GroupFeatureBuilder",
    "_BASE_SCALAR_DIM",
    "GroupSwitcher",
    "pairwise_logistic_ranking_loss",
    "hinge_ranking_loss",
    "listwise_softmax_loss",
    "build_pairs_from_scores",
    "build_pairs_from_ranking",
    "compute_ranking_accuracy",
    "compute_top1_accuracy",
    "RankingLossWithScheduledMargin",
    # RL
    "RLFeatureBuilder",
    "SwitcherActorCritic",
    "SwitcherPPO",
    "SwitcherRolloutBuffer",
    "SwitcherEnv",
    # Groups
    "generate_all_groups",
    "generate_original_groups",
    "generate_subgroups_recursive",
    "filter_groups_by_size",
    "print_group_statistics",
    "actions_for_group",
    "actions_for_group_from_raw",
    # Config
    "SwitcherScalarConfig",
    "load_switcher_config",
    "config_from_dict",
]
