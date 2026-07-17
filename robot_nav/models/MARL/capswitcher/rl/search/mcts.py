"""
Budgeted MCTS switcher: UCT tree search with the learned value replacing rollouts.

No rollout runs anywhere — a leaf's evaluation is ``model.cost_to_go`` (with a
configured ``leaf_value`` that is the learned per-robot precise cost-to-go,
summed over unreached robots).  With Bellman backup and a budget large enough to
expand every reachable node, the root decision equals the exhaustive minimin of
``rl/mpc/search.py``; smaller budgets spend expansions *asymmetrically* along
promising lines — that adaptivity vs. uniform-depth MPC is exactly what the
budget-matched comparison measures.

Budget = number of node expansions (each expansion runs the coarse vetting once
and one precise-all model rollout — the dominant costs), so ``budget=21`` is
directly comparable to exhaustive depth-3 minimin (<= 21 expansions).
"""

from __future__ import annotations

import math

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.mpc.forward_model import ForwardModel
from robot_nav.models.MARL.capswitcher.rl.mpc.mpc_switcher import MPCSwitcher
from robot_nav.models.MARL.capswitcher.rl.search.common import QNormalizer
from robot_nav.models.MARL.capswitcher.rl.search.tree import (
    Node,
    decision_dict,
    edge_value,
    expand_node,
    make_node,
    simulate,
)


class MCTS:
    """
    One search instance (fresh per real decision, receding horizon).

    Args:
        budget:  Max node expansions (the root expansion counts).
        backup:  ``"bellman"`` (value replacement — default, matches the
                 deterministic min-cost semantics) or ``"mean"`` (classic
                 averaging; winner's-curse ablation).
        c_uct:   Exploration constant on the min–max-normalised values.
    """

    def __init__(self, budget: int, backup: str = "bellman", c_uct: float = 1.0) -> None:
        if backup not in ("bellman", "mean"):
            raise ValueError(f"backup must be 'bellman' or 'mean', got {backup!r}")
        self.budget = int(budget)
        self.backup = backup
        self.c_uct = float(c_uct)

    def _select(self, root: Node):
        qnorm = QNormalizer()
        backup_mode = self.backup

        def select_action(node: Node, g: float) -> int:
            n_actions = len(node.branches)
            values = [edge_value(node, a, backup=backup_mode) for a in range(n_actions)]
            for v in values:
                qnorm.update(v)

            # Branch-and-bound mask: with positive step costs, an edge whose
            # exact root-path cost already reaches the root incumbent can never
            # beat it.  Only applied while at least one edge survives.
            allowed = [
                a for a in range(n_actions)
                if g + node.branches[a].step_cost < root.U
            ] or list(range(n_actions))

            unvisited = [a for a in allowed if node.N[a] == 0]
            if unvisited:
                # Visit unvisited edges greedily by their leaf-based estimate.
                return min(unvisited, key=lambda a: values[a])

            total = int(node.N[allowed].sum())
            log_total = math.log(max(total, 1))

            def score(a: int) -> float:
                explore = self.c_uct * math.sqrt(log_total / node.N[a])
                return qnorm.normalize(values[a]) + explore

            return max(allowed, key=score)

        return select_action

    def run(self, model: ForwardModel, ms) -> dict:
        root = make_node(model, ms)
        expand_node(root, model)
        expansions = 1
        select_action = self._select(root)

        # Descents that end on terminal/collision leaves expand nothing; the
        # attempt cap keeps the loop finite when the tree saturates.
        attempts, max_attempts = 0, 8 * self.budget
        while expansions < self.budget and attempts < max_attempts:
            expansions += simulate(
                root, model, select_action, backup_mode=self.backup
            )
            attempts += 1

        return decision_dict(root, backup_mode=self.backup)


class MCTSSwitcher(MPCSwitcher):
    """
    Drop-in switcher running budgeted MCTS instead of fixed-depth minimin.

    Shares the model construction (and the ``decide`` dict contract) with
    :class:`MPCSwitcher`; the inherited ``depth`` is unused.

    Args:
        budget: Node expansions per real decision.
        backup: ``"bellman"`` (default) or ``"mean"``.
        c_uct:  UCT exploration constant.
        (remaining kwargs as for ``MPCSwitcher``.)
    """

    def __init__(self, *args, budget: int = 21, backup: str = "bellman",
                 c_uct: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.search = MCTS(budget=budget, backup=backup, c_uct=c_uct)

    def decide(self, robot_state: np.ndarray) -> dict:
        model = self._build_model(robot_state)
        ms = ForwardModel.state_from_robot_state(robot_state)
        decision = self.search.run(model, ms)
        self.decision_expansions.append(model.n_precise_expansions)
        return decision
