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

from dataclasses import dataclass

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE
from robot_nav.models.MARL.capswitcher.rl.mpc.forward_model import (
    ForwardModel,
    ModelState,
)

# Dominating cost for a transition into a predicted collision — larger than any
# realistic cost-to-go, so the search shuns it without ever pruning to "no option".
COLLISION_COST: float = 1e9


@dataclass
class Branch:
    """One expanded action edge from a node."""

    mode: int                 # COARSE (0) or PRECISE (1)
    group: int | None         # coarse group id, else None
    frames: list | None       # exact vetted coarse frames to execute, else None
    child: ModelState         # deterministic next state
    step_cost: float          # per-decision motion cost of taking this action


def _expand(model: ForwardModel, ms: ModelState) -> tuple[list[Branch], list]:
    """
    Expand ``ms`` into its allowed action edges and return the coarse candidates.

    Coarse edges are included only for shield-safe groups; the precise-all edge is
    always included.  The candidate list (all selectable groups, safe or not) is
    returned for the decision dict / availability metrics.
    """
    moves = model.coarse_moves(ms)
    candidates = [mv.candidate for mv in moves.values()]

    branches: list[Branch] = []
    for group, mv in moves.items():
        if mv.candidate.safe:
            branches.append(
                Branch(
                    mode=COARSE,
                    group=group,
                    frames=mv.candidate.frames,
                    child=mv.next_state,
                    step_cost=model.step_cost(COARSE, group),
                )
            )
    branches.append(
        Branch(
            mode=PRECISE,
            group=None,
            frames=None,
            child=model.precise_next(ms),
            step_cost=model.step_cost(PRECISE),
        )
    )
    return branches, candidates


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
