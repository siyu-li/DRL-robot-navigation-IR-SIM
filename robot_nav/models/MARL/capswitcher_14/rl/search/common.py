"""
Shared machinery for the lazy tree search: branch stubs and value normalisation.

The economic premise (Danihelka et al., ICLR 2022) is an asymmetry: *ranking*
actions is cheap (one prior forward pass covers all edges), *evaluating* them
is expensive (a coarse vet or a precise rollout per edge).  Accordingly a
:class:`Branch` starts as a **stub** — mode, group and exact step cost, all
known without touching the model — and its transition is materialised only when
the search actually descends the edge.  Contrast ``capswitcher.rl.search.common``,
whose eager ``expand`` vets every group and rolls out precise at every node.

Step costs are exact at stub creation (member count × move distance / flat
precise cost), so selection can price edges before buying their transitions.

A shield-refuted coarse edge is an *illegal action discovered late*: it is
pruned from the node's legal set — no Q value, no collision penalty.  The
``COLLISION_COST`` sentinel survives only for **states** predicted to be in
collision (possible for precise children, which are not shield-vetted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE

# Dominating cost for a transition into a predicted-collision state — larger
# than any realistic cost-to-go, so the search shuns it without pruning.
COLLISION_COST: float = 1e9


@dataclass
class Branch:
    """
    One action edge from a node — created as a stub, filled on materialisation.

    ``frames`` / ``candidate`` stay ``None`` until the edge is materialised
    (and ``frames`` stays ``None`` for the precise edge, which has no vetted
    frame plan).  The child node lives on the owning ``Node.children`` slot.
    """

    mode: int                 # COARSE (0) or PRECISE (1)
    group: int | None         # coarse move-group id, else None
    step_cost: float          # exact per-decision motion cost (known at stub time)
    frames: list | None = field(default=None)     # vetted coarse frames (post-mat.)
    candidate: object | None = field(default=None)  # shield CoarseCandidate (post-mat.)


def expand_stubs(model, ms) -> list[Branch]:
    """
    Build the branch stubs for ``ms``: every selectable coarse group plus the
    always-present precise edge (last).  No model transition is run — this is
    the cheap half of an expansion; the prior forward pass is the other half.
    """
    branches = [
        Branch(mode=COARSE, group=g, step_cost=model.step_cost(COARSE, g))
        for g in model.coarse.selectable_groups()
    ]
    branches.append(Branch(mode=PRECISE, group=None, step_cost=model.step_cost(PRECISE)))
    return branches


class QNormalizer:
    """
    Running min–max normaliser mapping cost-to-go values to a higher-is-better
    q ∈ [0, 1] (the MuZero/Gumbel trick; keeps ``σ`` budget-independent).

    Collision-dominated costs (``>= COLLISION_COST``) never update the bounds
    and always normalise to 0 (worst).  Before two distinct finite values have
    been seen, every finite cost normalises to 0.5.
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
