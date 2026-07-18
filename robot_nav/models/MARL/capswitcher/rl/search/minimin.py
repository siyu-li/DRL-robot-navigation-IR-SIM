"""
Exhaustive fixed-depth minimin lookahead + its receding-horizon switcher (MPC).

The baseline search of the family in this package: uniform depth-``d`` expansion,
no budget adaptivity.  At each node the value is the cheapest way to drive every
robot home within the remaining horizon; below the horizon it is approximated by
the model's cost-to-go heuristic:

    V(x, d) = 0                                   if all robots reached
            = H(x)                                if d == 0
            = min_{a ∈ allowed(x)}  c(x,a) + V(T(x,a), d-1)

``allowed(x)`` = safety-masked coarse groups (shield) ∪ {precise-all}; precise is
always available, so a decision always exists.  Everything is deterministic, so
there are no chance nodes (this is minimin lookahead, not expectimax).  A transition
into a predicted collision is charged a dominating cost so the search avoids it while
still always having a defined value.

:class:`MPCSwitcher` wraps the recursion into the standard ``decide(robot_state)``
contract and doubles as the base class of the budgeted switchers
(``mcts.MCTSSwitcher`` / ``gumbel.GumbelSwitcher``), which reuse its model
construction and differ only in the search they run.
"""

from __future__ import annotations

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.forward_model import (
    ForwardModel,
    ModelState,
    build_forward_model,
)
from robot_nav.models.MARL.capswitcher.rl.reward import PathCostReward

# Expansion machinery shared with the MCTS / Gumbel searches.
from robot_nav.models.MARL.capswitcher.rl.search.common import (
    COLLISION_COST,
    Branch,
    expand,
)


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

    branches, _ = expand(model, ms)
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
    branches, candidates = expand(model, ms)

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


class MPCSwitcher:
    """
    Receding-horizon MPC switcher (fixed-depth minimin).

    Drop-in for the shielded policies — ``decide(robot_state)`` returns the same
    dict contract, so it plugs straight into
    ``SwitcherEnv.step(mode, group=..., frames=...)`` and the eval harness.  At
    each real decision it rebuilds the deterministic :class:`ForwardModel` from
    the live simulator (obstacles are frozen within the decision), runs the
    depth-d minimin lookahead, executes the first action, and re-plans on the
    next call (receding horizon).

    The root node is reconstructed from the passed ``robot_state`` (which is
    invertible back to poses + last action), so the plan is scored on exactly
    the state the frozen GAT produced its embeddings from.

    Args:
        backbone:           Frozen ``GATBackbone`` (shared with the env).
        coarse:             ``CoarseSteering`` primitive (shared with the env).
        sim:                ``MARL_SIM_OBSTACLE`` (live obstacle/geometry source).
        depth:              Lookahead depth in decisions (1 = myopic).
        d_safe:             Clearance margin (m) for the coarse safety mask.
        alpha:              Cost-to-go slope; ``None`` uses
                            ``precise_cost / (lin_max · step_time)``.
        selection_interval: Sub-steps per robot in a precise decision (match env).
        goal_threshold:     Per-robot goal-arrival radius (m) for the model's
                            ``all_reached`` terminal.
        reward_fn:          ``PathCostReward`` supplying the motion cost (defaults
                            to the eval configuration).
        default_rho:        Fallback robot radius if the sim does not expose one.
        leaf_value:         Optional learned leaf evaluator ``(model, ms) -> float``
                            (e.g. ``LearnedCostToGo``); replaces the crude
                            cost-to-go heuristic at the horizon.
    """

    def __init__(
        self,
        backbone,
        coarse,
        sim,
        depth: int = 2,
        d_safe: float = 0.3,
        alpha: float | None = None,
        selection_interval: int = 5,
        goal_threshold: float = 0.3,
        reward_fn: PathCostReward | None = None,
        default_rho: float = 0.2,
        leaf_value=None,
    ) -> None:
        self.backbone = backbone
        self.coarse = coarse
        self.sim = sim
        self.depth = int(depth)
        self.d_safe = float(d_safe)
        self.alpha = alpha
        self.selection_interval = int(selection_interval)
        self.goal_threshold = float(goal_threshold)
        self.reward_fn = reward_fn if reward_fn is not None else PathCostReward()
        self.default_rho = float(default_rho)
        self.leaf_value = leaf_value
        # Per-decision node-expansion counts (budget accounting for eval).
        self.decision_expansions: list[int] = []

    def _build_model(self, robot_state: np.ndarray) -> ForwardModel:
        """Rebuild the forward model from the current sim + passed root state."""
        return build_forward_model(
            self.backbone,
            self.coarse,
            self.sim,
            robot_state,
            d_safe=self.d_safe,
            selection_interval=self.selection_interval,
            goal_threshold=self.goal_threshold,
            reward_fn=self.reward_fn,
            default_rho=self.default_rho,
            leaf_value=self.leaf_value,
        )

    def decide(self, robot_state: np.ndarray) -> dict:
        """
        Return the receding-horizon decision for the current state.

        Returns:
            ``{"mode", "group", "frames", "candidates"}`` — for a coarse decision
            ``frames`` are the exact seeded frames the lookahead vetted, so the
            executed control equals the scored control.
        """
        model = self._build_model(robot_state)
        ms = ForwardModel.state_from_robot_state(robot_state)
        decision = plan_decision(model, ms, depth=self.depth, alpha=self.alpha)
        self.decision_expansions.append(model.n_precise_expansions)
        return {
            "mode": decision["mode"],
            "group": decision["group"],
            "frames": decision["frames"],
            "candidates": decision["candidates"],
        }
