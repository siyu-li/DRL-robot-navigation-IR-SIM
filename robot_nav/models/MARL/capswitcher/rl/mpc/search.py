"""
Depth-limited, deterministic min-cost lookahead (minimin) for the MPC switcher.

At each node the value is the cheapest way to drive every robot home within the
remaining horizon; below the horizon it is approximated by the model's cost-to-go
heuristic:

    V(x, d) = 0                                   if all robots reached
            = H(x)                                if d == 0
            = min_{a ∈ allowed(x)}  c(x,a) + V(T(x,a), d-1)

``allowed(x)`` = safety-masked coarse groups (shield) ∪ {precise-all}; precise is
always available, so a decision always exists.  Everything is deterministic, so
there are no chance nodes (this is minimin lookahead, not expectimax).  A transition
into a predicted collision is charged a dominating cost so the search avoids it while
still always having a defined value.
"""

from __future__ import annotations

from robot_nav.models.MARL.capswitcher.rl.mpc.forward_model import (
    ForwardModel,
    ModelState,
)

# Expansion machinery shared with the MCTS / Gumbel searches — re-exported here
# so existing imports of ``search.Branch`` / ``search.COLLISION_COST`` keep working.
from robot_nav.models.MARL.capswitcher.rl.search.common import (  # noqa: F401
    COLLISION_COST,
    Branch,
    expand,
)

_expand = expand


def _branch_value(
    model: ForwardModel, branch: Branch, depth: int, alpha: float | None
) -> float:
    """Total cost of taking ``branch`` then acting optimally for ``depth`` more."""
    if model.collision_pred(branch.child):
        return COLLISION_COST
    return branch.step_cost + _value(model, branch.child, depth, alpha)


def _value(
    model: ForwardModel, ms: ModelState, depth: int, alpha: float | None
) -> float:
    """Minimin value of ``ms`` with ``depth`` decisions of lookahead remaining."""
    if model.all_reached(ms):
        return 0.0
    if depth <= 0:
        return model.cost_to_go(ms, alpha)

    branches, _ = _expand(model, ms)
    return min(_branch_value(model, b, depth - 1, alpha) for b in branches)


def plan_decision(
    model: ForwardModel,
    ms: ModelState,
    depth: int,
    alpha: float | None = None,
) -> dict:
    """
    Choose the best action at ``ms`` by ``depth``-step lookahead.

    Args:
        model: the analytic forward model.
        ms:    root state.
        depth: lookahead depth in decisions (``depth=1`` is the myopic special
               case: minimise ``cost + cost_to_go(child)``).
        alpha: cost-to-go slope; ``None`` uses the model default.

    Returns:
        Decision dict matching the shielded-policy contract:
        ``{"mode", "group", "frames", "candidates", "value"}``.  For a coarse
        decision ``frames`` are the exact seeded frames that were vetted (so the
        executed plan equals the scored plan).
    """
    branches, candidates = _expand(model, ms)

    best: Branch | None = None
    best_value = float("inf")
    for b in branches:
        value = _branch_value(model, b, depth - 1, alpha)
        if value < best_value:
            best_value, best = value, b

    return {
        "mode": best.mode,
        "group": best.group,
        "frames": best.frames,
        "candidates": candidates,
        "value": best_value,
    }
