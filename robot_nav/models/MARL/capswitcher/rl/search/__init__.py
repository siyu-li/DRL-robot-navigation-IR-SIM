"""
Budgeted tree searches over the analytic forward model (CAPSwitcher).

Three switchers share the same deterministic model, expansion machinery, leaf
value, and decision contract — they differ only in how they allocate a fixed
node-expansion budget:

  * ``rl/mpc``            — exhaustive fixed-depth minimin (uniform tree).
  * ``search.mcts``       — UCT MCTS, learned value instead of rollouts.
  * ``search.gumbel``     — Gumbel AlphaZero planning with a base-policy prior.

Only the dependency-free building blocks are exported at package level;
import ``MCTSSwitcher`` / ``GumbelSwitcher`` from their submodules
(``rl.search.mcts`` / ``rl.search.gumbel``) — importing them here would be
circular, because ``rl.mpc`` itself pulls the shared expansion machinery from
``rl.search.common``.
"""

from robot_nav.models.MARL.capswitcher.rl.search.common import (
    COLLISION_COST,
    Branch,
    QNormalizer,
    build_forward_model,
    expand,
)
from robot_nav.models.MARL.capswitcher.rl.search.priors import (
    HeuristicPrior,
    UniformPrior,
)

__all__ = [
    "COLLISION_COST",
    "Branch",
    "QNormalizer",
    "build_forward_model",
    "expand",
    "HeuristicPrior",
    "UniformPrior",
]
