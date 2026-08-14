"""
Eager search tree for the 14-robot Gumbel AlphaZero switcher — **GAZ14-E**.

The sibling of ``tree.py`` (GAZ14-L, lazy), and the 14-robot restatement of the
6-robot ``capswitcher.rl.search.tree``: *expanding a node means buying every
edge*.  All 22 coarse groups are shield-vetted and the precise edge is rolled
out, so every legal edge of an expanded node carries a real Q value.

What that buys, and what it costs
---------------------------------
* **No completed-Q.**  ``tree.py`` needs the mixed value estimate ``v_mix``
  because most of its edges are never materialised.  Here there is nothing to
  complete: the policy over edges is ``softmax(logits + σ(q))`` on real
  values.  The pessimistic-leaf failure mode of the lazy rule is confined to
  *nodes* the search never expands, never to edges it never priced.
* **No advisory feasibility mask.**  Every coarse edge is really vetted at
  every node, so a *predicted* feasibility mask has nothing left to do.
* **The prior may read the vet.**  ``expand_node_eager`` calls the prior
  **after** materialising, so branches carry ``candidate`` (clearance,
  progress) and ``progress``.  This is exactly the dependence the lazy prior
  is forbidden (see ``priors.py``) — there the vet is the cost the prior
  exists to avoid; here it is already sunk.  See ``priors_eager.py``.
* **The price is breadth.**  One expansion costs
  :func:`expansion_cost` = 22 vets + 1 rollout = 23 transitions, *always*
  (``selectable_groups()`` is static; robots reaching their goals do not
  shrink the group table).  A budget therefore buys ``budget // 23`` nodes,
  against the lazy tree's ``budget`` single edges.  Which side of that trade
  wins is the experiment the two variants exist to run.

Fixed-width branch layout
-------------------------
Unlike the 6-robot eager tree — which drops refuted coarse groups from the
branch list — a node here keeps all 23 branch slots and marks a refuted coarse
edge ``legal[a] = False`` (the lazy tree's pruning semantics: an illegal action
discovered late; no Q, no collision penalty).  That keeps the root's π′ vector
and the harvested clearance labels shape-identical to GAZ14-L's, so
``features.py`` / ``prior_net.py`` / ``train_prior.py`` and the logged shards
are shared verbatim between the two variants and their teachers are directly
comparable.

Node representation, ``edge_value``, Bellman refresh with certificates, and
``backup`` are imported unchanged from ``tree.py``: on a fully expanded node
"materialised" and "legal" coincide, so those operators already do the right
thing.
"""

from __future__ import annotations

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE
from robot_nav.models.MARL.capswitcher_14.rl.search.common import expand_stubs
from robot_nav.models.MARL.capswitcher_14.rl.search.tree import (
    Node,
    backup,
    bellman_refresh,
    make_node,
    materialize,
)

__all__ = [
    "Node",
    "make_node",
    "expansion_cost",
    "expand_node_eager",
    "simulate_eager",
]


def expansion_cost(model) -> int:
    """
    Transitions one eager node expansion costs: one vet per selectable coarse
    group plus the precise rollout.

    Constant over the whole search (the move-group table is static), which is
    what makes the eager budget exactly ``expansion_cost × nodes`` and lets the
    caller decide *before* a descent whether it can afford one.
    """
    return len(model.coarse.selectable_groups()) + 1


def _precise_progress(model, ms, child_ms) -> float | None:
    """
    Goal-distance reduction the precise rollout achieved, summed over the
    robots that were unreached at ``ms``.

    Free — the rollout has already been paid for — and it puts the precise edge
    on the same progress-per-cost currency as the coarse candidates, which is
    what lets :class:`priors_eager.HeuristicPrior14` score all 23 edges
    together instead of reaching for a hand-tuned precise bias.

    Returns ``None`` for models that expose no ``goal_distances`` (the unit-test
    stubs); the prior then treats the precise edge as median-efficient.
    """
    goal_distances = getattr(model, "goal_distances", None)
    if goal_distances is None:
        return None
    d0 = np.asarray(goal_distances(ms), dtype=np.float64)
    d1 = np.asarray(goal_distances(child_ms), dtype=np.float64)
    unreached = d0 > float(getattr(model, "goal_threshold", 0.0))
    return float(np.sum(d0[unreached] - d1[unreached]))


def expand_node_eager(node: Node, model, prior) -> None:
    """
    Expand ``node`` by buying every edge: vet all coarse groups, roll out
    precise, leaf-evaluate the children, then rank with the prior and take the
    node's first Bellman refresh.

    Costs exactly :func:`expansion_cost` transitions — refuted vets included,
    since a refutation consumed the same sweep computation as an admission.
    The prior runs **last** so it can read the vet results (see the module
    docstring); ``Branch.progress`` is filled here for both edge kinds.
    """
    node.branches = expand_stubs(model, node.ms)
    k = len(node.branches)
    node.children = [None] * k
    node.legal = np.ones(k, dtype=bool)
    node.N = np.zeros(k, dtype=np.int64)
    node.W = np.zeros(k, dtype=np.float64)

    for a in range(k):
        child = materialize(node, a, model)      # prunes the edge if refuted
        branch = node.branches[a]
        if branch.mode == COARSE:
            if branch.candidate is not None:
                branch.progress = float(branch.candidate.progress)
        elif child is not None:
            branch.progress = _precise_progress(model, node.ms, child.ms)

    node.prior_logits = np.asarray(
        prior(model, node.ms, node.branches), dtype=np.float64
    )
    bellman_refresh(node)


def simulate_eager(
    root: Node,
    model,
    select_action,
    prior,
    first_action: int | None = None,
) -> int:
    """
    One descent: select down from ``root`` to an unexpanded node, expand it
    (at most one expansion), back up.

    ``select_action(node, g)`` picks a legal edge given the exact accumulated
    path cost ``g``; on an eagerly expanded node every legal edge already has a
    child, so a descent never has to buy anything mid-path.  ``first_action``
    forces the root edge (Sequential Halving).  Descents that end on a terminal
    or predicted-collision node expand nothing but still back up their visit,
    so selection statistics move on.

    The caller is responsible for checking it can afford :func:`expansion_cost`
    before calling — expansions are atomic and must never be half-bought.

    Returns:
        The number of node expansions performed (0 or 1).
    """
    path: list[tuple[Node, int]] = []
    node = root
    g = 0.0

    if first_action is not None:
        child = root.children[first_action]
        if child is None:                        # refuted arm — nothing to do
            return 0
        path.append((root, first_action))
        g += root.branches[first_action].step_cost
        node = child

    expansions = 0
    while True:
        if node.terminal or node.collision:
            break
        if not node.expanded:
            expand_node_eager(node, model, prior)
            expansions = 1
            break
        a = select_action(node, g)
        child = node.children[a]
        if child is None:                        # selection must stay legal
            raise AssertionError(
                f"select_action returned unmaterialised edge {a} "
                f"(legal={node.legal.tolist()})"
            )
        path.append((node, a))
        g += node.branches[a].step_cost
        node = child

    backup(path, node)
    return expansions
