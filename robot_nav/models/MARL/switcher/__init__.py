"""
Group Switcher Module for Ranking-Based Group Selection.

This module provides:
- GroupFeatureBuilder: Constructs feature vectors for candidate groups
- GroupSwitcher: MLP network for scoring and selecting groups
- Ranking losses for training (pairwise logistic, hinge)
- Training utilities (SwitcherDataset, SwitcherTrainer)

Integration with existing workspace:
- Uses per-robot embeddings from GAT backbone (iga_obstacle.py)
- Uses attention weights from AttentionObstacle (hard_weights_rr, hard_weights_ro)
- Compatible with existing group definitions from group_switch_planner.py

Training:
    # 1. Collect oracle data
    python -m robot_nav.models.MARL.switcher.collect_oracle_data --output_path data/oracle_data.pt
    
    # 2. Train switcher
    python -m robot_nav.models.MARL.switcher.train_switcher --data_path data/oracle_data.pt
"""

from robot_nav.models.MARL.switcher.feature_builder import (
    GroupFeatureBuilder,
    compute_attention_stats_vectorized,
)
from robot_nav.models.MARL.switcher.switcher_net import (
    GroupSwitcher,
    GroupSwitcherWithBaseline,
)
from robot_nav.models.MARL.switcher.rank_losses import (
    pairwise_logistic_ranking_loss,
    hinge_ranking_loss,
    listwise_softmax_loss,
    build_pairs_from_scores,
    build_pairs_from_ranking,
    compute_ranking_accuracy,
    compute_top1_accuracy,
    RankingLossWithScheduledMargin,
)

__all__ = [
    # Feature builder
    "GroupFeatureBuilder",
    "compute_attention_stats_vectorized",
    # Switcher networks
    "GroupSwitcher",
    "GroupSwitcherWithBaseline",
    # Ranking losses
    "pairwise_logistic_ranking_loss",
    "hinge_ranking_loss",
    "listwise_softmax_loss",
    "build_pairs_from_scores",
    "build_pairs_from_ranking",
    "compute_ranking_accuracy",
    "compute_top1_accuracy",
    "RankingLossWithScheduledMargin",
]
