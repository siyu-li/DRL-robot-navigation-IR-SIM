"""
Shared tree machinery for the budgeted searches (MCTS / Gumbel AlphaZero).

The problem is a *deterministic, known-model, min-cost* search: transitions come
from the analytic :class:`ForwardModel`, step costs are exact, and the only noise
is the leaf cost-to-go estimate (the learned ``v_psi`` or the analytic
``alpha * sum(dist)``).  That drives three design choices:

* **Bellman (value-replacement) backup is the default.**  The leaf value
  estimates the cost of the *precise-only* completion — an always-available
  fallback policy — so it upper-bounds the optimal cost-to-go in expectation.
  Replacing a node's value with ``min_a (c_a + V(child_a))`` on expansion swaps
  "estimated precise finish" for "best planned decision + estimated finish",
  a strictly more informed quantity (the classic rollout-improvement argument).
  ``backup="mean"`` (classic averaging) is kept as an ablation: min over noisy
  estimates is biased low (winner's curse), and which effect dominates with a
  given value net is an empirical question.

* **Certificates are stored separately from estimates.**  When a descent reaches
  an ``all_reached`` state, the suffix cost along the path is *exact in-model*:
  every node on the path gets a certified upper bound ``U`` (cost of a concrete,
  shield-vetted plan to the goal).  The effective node value is
  ``min(q_hat, U)`` — an estimate can never talk the search out of a cheaper
  plan it has already proven, while an estimate below ``U`` still invites
  exploration with bounded downside.  This is the one-player min-cost analogue
  of MCTS-Solver / score-bounded MCTS.

* **Positive step costs give a free branch-and-bound prune.**  Remaining cost is
  >= 0, so any edge whose exact accumulated root-path cost already reaches the
  root's best certified total can never improve the incumbent; selection masks
  such edges (unless nothing else is left).

With full expansion the Bellman-backed root value equals the minimin recursion of
``rl/mpc/search.py`` exactly — the budgeted searches differ only in *which* nodes
they choose to expand.
"""

from __future__ import annotations

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.search.common import (
    COLLISION_COST,
    Branch,
    QNormalizer,
    expand,
)

INF = float("inf")


class Node:
    """One search-tree node over a :class:`ModelState`."""

    __slots__ = (
        "ms", "terminal", "collision", "h", "q_hat", "U",
        "branches", "children", "candidates", "N", "W", "prior_logits",
    )

    def __init__(self, ms, terminal: bool, collision: bool, h: float) -> None:
        self.ms = ms
        self.terminal = terminal
        self.collision = collision
        self.h = h                     # leaf cost-to-go estimate at creation
        self.q_hat = h                 # Bellman estimate (refined on expansion)
        self.U = 0.0 if terminal else INF  # certified upper bound on cost-to-go
        self.branches: list[Branch] | None = None
        self.children: list[Node] | None = None
        self.candidates: list | None = None
        self.N: np.ndarray | None = None   # per-action visit counts
        self.W: np.ndarray | None = None   # per-action return sums (mean backup)
        self.prior_logits: np.ndarray | None = None

    @property
    def expanded(self) -> bool:
        return self.branches is not None

    def value(self) -> float:
        """Effective cost-to-go: the estimate capped by the certificate."""
        return min(self.q_hat, self.U)


def make_node(model, ms) -> Node:
    """Create (and leaf-evaluate) a node for ``ms``."""
    if model.all_reached(ms):
        return Node(ms, terminal=True, collision=False, h=0.0)
    if model.collision_pred(ms):
        return Node(ms, terminal=False, collision=True, h=COLLISION_COST)
    return Node(ms, terminal=False, collision=False, h=float(model.cost_to_go(ms)))


def edge_value(node: Node, a: int, backup: str = "bellman") -> float:
    """
    Cost of taking edge ``a`` from ``node`` then acting well below it.

    A collision child is charged the dominating ``COLLISION_COST`` (matching the
    minimin recursion).  Otherwise the estimate (Bellman child value, or the
    empirical mean return under ``backup="mean"``) is capped by the certificate
    ``step_cost + child.U``.
    """
    child = node.children[a]
    if child.collision:
        return COLLISION_COST
    cert = node.branches[a].step_cost + child.U
    if backup == "mean" and node.N[a] > 0:
        est = node.W[a] / node.N[a]
    else:
        est = node.branches[a].step_cost + child.q_hat
    return min(est, cert)


def _bellman_refresh(node: Node) -> None:
    """Recompute ``q_hat`` / ``U`` of an expanded node from its children."""
    node.q_hat = min(
        edge_value(node, a, backup="bellman") for a in range(len(node.branches))
    )
    node.U = min(
        (
            node.branches[a].step_cost + node.children[a].U
            for a in range(len(node.branches))
            if not node.children[a].collision
        ),
        default=INF,
    )


def expand_node(node: Node, model, prior=None) -> None:
    """
    Expand ``node``: build all action edges, leaf-evaluate every child, and do
    the node's first Bellman refresh (a one-step improvement over ``h``).

    All children are created here because :func:`expand` computes their states
    anyway — the expansion itself is the budgeted resource, child leaf
    evaluations are comparatively cheap.
    """
    branches, candidates = expand(model, node.ms)
    node.branches = branches
    node.candidates = candidates
    node.children = [make_node(model, b.child) for b in branches]
    node.N = np.zeros(len(branches), dtype=np.int64)
    node.W = np.zeros(len(branches), dtype=np.float64)
    if prior is not None:
        node.prior_logits = np.asarray(prior(model, node.ms, branches), dtype=np.float64)
    _bellman_refresh(node)


def backup(path: list[tuple[Node, int]], leaf: Node, mode: str = "bellman") -> None:
    """
    Back the descent result up ``path`` (root-first list of (node, action)).

    Certificates ``U`` propagate exactly in both modes.  ``bellman`` recomputes
    each node's ``q_hat`` from its children (value replacement); ``mean``
    accumulates the return ``G = sum of exact step costs + leaf effective value``
    into per-action running means.
    """
    G = leaf.value()
    for node, a in reversed(path):
        node.N[a] += 1
        G = node.branches[a].step_cost + G
        node.W[a] += G
        if mode == "bellman":
            _bellman_refresh(node)
        else:
            # Certificates still propagate exactly under mean backup.
            node.U = min(
                (
                    node.branches[b].step_cost + node.children[b].U
                    for b in range(len(node.branches))
                    if not node.children[b].collision
                ),
                default=INF,
            )


def simulate(
    root: Node,
    model,
    select_action,
    *,
    backup_mode: str = "bellman",
    prior=None,
    first_action: int | None = None,
) -> int:
    """
    One descent: select down from ``root``, expand one unexpanded leaf, back up.

    ``select_action(node, g)`` picks an edge at an expanded node given the exact
    accumulated path cost ``g`` (for the branch-and-bound mask).  Descents that
    end on a terminal / collision node expand nothing (return 0) but still back
    up their visit so selection statistics move on.

    Returns the number of node expansions performed (0 or 1).
    """
    path: list[tuple[Node, int]] = []
    node = root
    g = 0.0
    if first_action is not None:
        path.append((root, first_action))
        g += root.branches[first_action].step_cost
        node = root.children[first_action]

    expansions = 0
    while True:
        if node.terminal or node.collision:
            break
        if not node.expanded:
            expand_node(node, model, prior=prior)
            expansions = 1
            break
        a = select_action(node, g)
        path.append((node, a))
        g += node.branches[a].step_cost
        node = node.children[a]

    backup(path, node, mode=backup_mode)
    return expansions


def decision_dict(
    root: Node,
    backup_mode: str = "bellman",
    extras: dict | None = None,
    action: int | None = None,
) -> dict:
    """
    Build the standard switcher decision dict from a searched root.

    By default the chosen action minimises the effective edge value (estimate
    capped by certificate); ``action`` overrides it (the Gumbel search returns
    its sequential-halving winner instead).  For a coarse choice the returned
    frames are the exact vetted frames of the root branch, so the executed
    control equals the scored one.
    """
    values = [edge_value(root, a, backup=backup_mode) for a in range(len(root.branches))]
    best = int(np.argmin(values)) if action is None else int(action)
    b = root.branches[best]
    out = {
        "mode": b.mode,
        "group": b.group,
        "frames": b.frames,
        "candidates": root.candidates,
        "value": float(values[best]),
    }
    if extras:
        out.update(extras)
    return out
