"""
Shared expansion machinery for every tree search over the analytic forward model.

The minimin MPC (``rl/mpc/search.py``), the MCTS switcher (``rl/search/mcts.py``)
and the Gumbel AlphaZero switcher (``rl/search/gumbel.py``) all expand nodes the
same way: shield-safe coarse groups ∪ {precise-all}, with the precise edge always
present.  :class:`Branch` / :func:`expand` live here so the three searches rank
exactly the same action edges.

Also here:

* :class:`QNormalizer` — running min–max normalisation of edge costs into a
  higher-is-better q ∈ [0, 1] (the MuZero/Gumbel trick; both UCT and the Gumbel
  ``σ`` transform need value scales to be budget-independent).
* :func:`build_forward_model` — the one place a switcher turns (backbone, coarse,
  live sim, root robot_state) into a :class:`ForwardModel`, shared by
  ``MPCSwitcher`` and the tree-search switchers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE, PathCostReward
from robot_nav.models.MARL.capswitcher.rl.shield import ShieldGeometry

# NOTE: ``ForwardModel`` / ``ModelState`` are only type annotations here; importing
# them at module level would be circular (mpc.search re-exports from this module),
# so ``build_forward_model`` imports ForwardModel lazily.

# Dominating cost for a transition into a predicted collision — larger than any
# realistic cost-to-go, so a search shuns it without ever pruning to "no option".
COLLISION_COST: float = 1e9

# robot_state goal columns (see CoarseSteering.compute_actions docstring).
_GX, _GY = 9, 10


@dataclass
class Branch:
    """One expanded action edge from a node."""

    mode: int                 # COARSE (0) or PRECISE (1)
    group: int | None         # coarse group id, else None
    frames: list | None       # exact vetted coarse frames to execute, else None
    child: "ModelState"       # deterministic next state
    step_cost: float          # per-decision motion cost of taking this action
    candidate: object | None = None  # shield CoarseCandidate for coarse edges


def expand(model, ms) -> tuple[list[Branch], list]:
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
                    candidate=mv.candidate,
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


class QNormalizer:
    """
    Running min–max normaliser mapping edge costs to a higher-is-better
    q ∈ [0, 1].

    Collision-dominated costs (``>= COLLISION_COST``) are never used to update
    the bounds (they would squash every real value into a point) and always
    normalise to 0 (worst).  Before two distinct finite values have been seen,
    every finite cost normalises to 0.5.
    """

    def __init__(self) -> None:
        self.vmin = float("inf")
        self.vmax = float("-inf")

    def update(self, cost: float) -> None:
        if cost >= COLLISION_COST:
            return
        self.vmin = min(self.vmin, cost)
        self.vmax = max(self.vmax, cost)

    def normalize(self, cost: float) -> float:
        if cost >= COLLISION_COST:
            return 0.0
        if not (self.vmax > self.vmin):
            return 0.5
        q = (self.vmax - cost) / (self.vmax - self.vmin)
        return float(min(max(q, 0.0), 1.0))


def build_forward_model(
    backbone,
    coarse,
    sim,
    robot_state: np.ndarray,
    *,
    d_safe: float,
    selection_interval: int,
    goal_threshold: float,
    reward_fn: PathCostReward,
    default_rho: float,
    leaf_value=None,
):
    """Rebuild the deterministic forward model from the live sim + root state."""
    from robot_nav.models.MARL.capswitcher.rl.mpc.forward_model import ForwardModel

    s = np.asarray(robot_state, dtype=np.float64)
    goals = s[:, [_GX, _GY]]
    return ForwardModel(
        backbone=backbone,
        coarse=coarse,
        goals=goals,
        obstacle_states=sim.get_obstacle_states(),
        geom=ShieldGeometry.from_sim(sim, default_rho=default_rho),
        step_time=coarse.step_time,
        selection_interval=selection_interval,
        lin_max=coarse.lin_max,
        d_safe=d_safe,
        goal_threshold=goal_threshold,
        reward_fn=reward_fn,
        leaf_value=leaf_value,
    )
