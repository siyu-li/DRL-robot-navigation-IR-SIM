"""
Tree searches over the analytic forward model (CAPSwitcher).

All planners share the same deterministic model (``rl/forward_model.py``),
expansion machinery, leaf value, and ``decide`` dict contract — they differ only
in how they allocate node expansions:

  * ``minimin`` — exhaustive fixed-depth minimin (uniform tree; the MPC baseline)
    and :class:`MPCSwitcher`, the base class of the budgeted switchers.
  * ``mcts``    — UCT MCTS, learned value instead of rollouts.
  * ``gumbel``  — Gumbel AlphaZero planning with a base-policy prior.

Layering (strictly one-directional, no cycles):

    reward/shield -> common -> forward_model -> tree/minimin -> mcts/gumbel
"""

from robot_nav.models.MARL.capswitcher.rl.search.common import (
    COLLISION_COST,
    Branch,
    QNormalizer,
    expand,
)
from robot_nav.models.MARL.capswitcher.rl.search.gumbel import GumbelSwitcher
from robot_nav.models.MARL.capswitcher.rl.search.mcts import MCTSSwitcher
from robot_nav.models.MARL.capswitcher.rl.search.minimin import (
    MPCSwitcher,
    plan_decision,
)
from robot_nav.models.MARL.capswitcher.rl.search.priors import (
    HeuristicPrior,
    UniformPrior,
)

__all__ = [
    "COLLISION_COST",
    "Branch",
    "QNormalizer",
    "expand",
    "plan_decision",
    "MPCSwitcher",
    "MCTSSwitcher",
    "GumbelSwitcher",
    "HeuristicPrior",
    "UniformPrior",
]
