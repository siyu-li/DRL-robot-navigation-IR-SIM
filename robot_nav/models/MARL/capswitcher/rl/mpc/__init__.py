"""
Fixed-depth receding-horizon MPC switcher for the CAPSwitcher.

The inner solve is a depth-limited, deterministic min-cost tree search with a
heuristic leaf value (minimin lookahead) over a standalone analytic pose-model —
NOT expectimax: the model is deterministic (seeded coarse column-drop, deterministic
precise rollout), so there are no chance nodes.

  * ``forward_model`` — analytic transition model (coarse via the shield's exact
    swept-path reconstruction, precise via frozen-GAT-forward + unicycle integration).
  * ``search``        — the ``V(x, d)`` recursion, cost-to-go heuristic, safety mask.
  * ``mpc_switcher``  — ``MPCSwitcher.decide`` with the same dict contract as the
    shielded policies, so it drops into ``SwitcherEnv.step`` and the eval harness.
"""

from robot_nav.models.MARL.capswitcher.rl.mpc.forward_model import (
    ForwardModel,
    ModelState,
)
from robot_nav.models.MARL.capswitcher.rl.mpc.search import plan_decision
from robot_nav.models.MARL.capswitcher.rl.mpc.mpc_switcher import MPCSwitcher

__all__ = [
    "ForwardModel",
    "ModelState",
    "plan_decision",
    "MPCSwitcher",
]
