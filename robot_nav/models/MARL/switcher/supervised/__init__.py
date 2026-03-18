"""
Supervised (oracle-based) group switcher components.

- ``GroupFeatureBuilder``: constructs feature vectors for candidate groups.
- ``GroupSwitcher``: two-tower fusion network that scores/ranks groups.
- Ranking losses: pairwise logistic, hinge, listwise softmax, scheduled margin.
"""

from robot_nav.models.MARL.switcher.supervised.feature_builder import (
    GroupFeatureBuilder,
    _BASE_SCALAR_DIM,
)
from robot_nav.models.MARL.switcher.supervised.switcher_net import (
    GroupSwitcher,
)
from robot_nav.models.MARL.switcher.supervised.rank_losses import (
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
]
