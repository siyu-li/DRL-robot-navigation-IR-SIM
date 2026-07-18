"""
Shared expansion machinery for every tree search over the analytic forward model.

The minimin MPC (``rl/search/minimin.py``), the MCTS switcher
(``rl/search/mcts.py``) and the Gumbel AlphaZero switcher (``rl/search/gumbel.py``)
all expand nodes the same way: shield-safe coarse groups U {precise-all}, with the
precise edge always present.  :class:`Branch` / :func:`expand` live here so the
three searches rank exactly the same action edges.

Also here: :class:`QNormalizer` — running min–max normalisation of edge costs
into a higher-is-better q ∈ [0, 1] (the MuZero/Gumbel trick; both UCT and the
Gumbel ``σ`` transform need value scales to be budget-independent).

This module is deliberately a dependency leaf (only ``reward``): ``Branch.child``
stays a string annotation so no model import is needed, and the model factory
lives with the model in ``rl/forward_model.py::build_forward_model``.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE

# Dominating cost for a transition into a predicted collision — larger than any
# realistic cost-to-go, so a search shuns it without ever pruning to "no option".
COLLISION_COST: float = 1e9


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
