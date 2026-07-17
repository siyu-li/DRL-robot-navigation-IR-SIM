"""
Gumbel AlphaZero switcher — planning with Gumbel over the exact analytic model.

This is the *planning-only* half of "Policy Improvement by Planning with Gumbel"
(Danihelka et al., ICLR 2022): Gumbel AlphaZero, i.e. the variant **without
model learning** — our hand-built :class:`ForwardModel` plays the role MuZero's
learned dynamics network exists to fill (the un-branchable irsim), except it is
exact.  No network is trained here; the base policy is a pluggable prior
(:class:`HeuristicPrior` for now), and the improved root policy ``pi'`` is
returned on every decision as the future distillation target.

Procedure (paper's Algorithm 2, adapted to min-cost):

* **Root**: sample Gumbel noise ``gum(a)``, keep the top-m actions by
  ``gum + logits``, then run Sequential Halving (Karnin et al. 2013): visit the
  survivors equally, halve by ``gum(a) + logits(a) + sigma(q_hat(a))``, and act
  with the final survivor.  ``q_hat`` is the min–max-normalised *negated* cost
  (higher is better), ``sigma(q) = (c_visit + max_b N(b)) * c_scale * q``.
* **Non-root**: the deterministic rule — build the improved policy
  ``pi' = softmax(logits + sigma(q))`` and select
  ``argmax_a pi'(a) - N(a)/(1 + sum_b N(b))`` (Eq. 14).  One simplification over
  the paper: completed Q-values exist because MuZero cannot cheaply evaluate
  unvisited actions, but our expansion leaf-evaluates *every* child, so all
  actions have value estimates and no completion placeholder is needed.
* **Backup**: Bellman value replacement with certificates, shared with the MCTS
  switcher (see ``tree.py`` for why that is the right operator here).

A "visit" descends the chosen root subtree once and expands one new node, so the
i.i.d.-bandit reading of Sequential Halving is approximate — as in Go/chess, it
acts as a principled budget allocator across root actions.

Known property (inherent to the AlphaZero family, not a defect): below the
root, the deterministic rule is *exploitative* — with the paper's ``sigma``
scaling, ``pi'`` is near-argmax, so a subtree whose leaf estimate is
pessimistic (overestimates its cost) may never be expanded and its value never
corrected downward.  Optimistic leaf errors self-correct (the edge gets
expanded and its backed-up cost rises); pessimistic ones persist.  Root arms
are protected by Sequential Halving's forced balanced visits.  The MCTS
switcher's UCT bonus explores regardless — one more axis the budget-matched
comparison measures.
"""

from __future__ import annotations

import math

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.mpc.forward_model import (
    ForwardModel,
    _state_group_seed,
)
from robot_nav.models.MARL.capswitcher.rl.mpc.mpc_switcher import MPCSwitcher
from robot_nav.models.MARL.capswitcher.rl.search.common import QNormalizer
from robot_nav.models.MARL.capswitcher.rl.search.priors import HeuristicPrior
from robot_nav.models.MARL.capswitcher.rl.search.tree import (
    Node,
    decision_dict,
    edge_value,
    expand_node,
    make_node,
    simulate,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


class GumbelAlphaZero:
    """
    One search instance (fresh per real decision, receding horizon).

    Args:
        prior:        Base policy ``prior(model, ms, branches) -> logits``.
        budget:       Max node expansions (the root expansion counts).
        m:            Max root actions sampled without replacement (<= k; with
                      k <= 4 legal actions this is usually all of them).
        c_visit, c_scale: The ``sigma`` transform constants (paper defaults).
        gumbel_scale: Scale of the root Gumbel noise; 0 makes the root
                      deterministic (used by the equivalence tests).
        seed:         Mixed with a state hash per decision, so a given root
                      state plans reproducibly.
    """

    def __init__(
        self,
        prior,
        budget: int,
        m: int = 4,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        gumbel_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.prior = prior
        self.budget = int(budget)
        self.m = int(m)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self.gumbel_scale = float(gumbel_scale)
        self.seed = int(seed)

    def _sigma(self, q: np.ndarray, root: Node) -> np.ndarray:
        return (self.c_visit + float(root.N.max())) * self.c_scale * q

    def run(self, model: ForwardModel, ms) -> dict:
        root = make_node(model, ms)
        expand_node(root, model, prior=self.prior)
        expansions = 1
        k = len(root.branches)
        logits = root.prior_logits

        rng = np.random.default_rng(
            (_state_group_seed(ms.poses, 0) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        )
        gum = rng.gumbel(size=k) * self.gumbel_scale

        qnorm = QNormalizer()

        def q_of(node: Node) -> np.ndarray:
            """Normalised (higher-better) q per edge of an expanded node."""
            values = [edge_value(node, a) for a in range(len(node.branches))]
            for v in values:
                qnorm.update(v)
            return np.array([qnorm.normalize(v) for v in values])

        def select_action(node: Node, g: float) -> int:
            n_actions = len(node.branches)
            allowed = [
                a for a in range(n_actions)
                if g + node.branches[a].step_cost < root.U
                and not node.children[a].collision
            ] or list(range(n_actions))
            pi = _softmax(node.prior_logits + self._sigma(q_of(node), node))
            total = float(node.N.sum())
            gap = pi - node.N / (1.0 + total)
            return max(allowed, key=lambda a: gap[a])

        def root_score(a: int) -> float:
            v = edge_value(root, a)
            qnorm.update(v)
            sig = self._sigma(np.array([qnorm.normalize(v)]), root)[0]
            return gum[a] + logits[a] + sig

        m = min(self.m, k)
        active = sorted(range(k), key=lambda a: gum[a] + logits[a], reverse=True)[:m]
        # Sequential Halving over the active arms; with a single legal arm the
        # loop degenerates to spending the whole budget refining its subtree.
        phases = max(1, math.ceil(math.log2(max(m, 2))) if m > 1 else 1)
        n = max(self.budget - 1, 0)
        for _ in range(phases):
            if expansions >= self.budget:
                break
            visits = max(1, n // (phases * len(active)))
            for a in active:
                for _ in range(visits):
                    if expansions >= self.budget:
                        break
                    expansions += simulate(
                        root, model, select_action,
                        backup_mode="bellman", prior=self.prior,
                        first_action=a,
                    )
            if len(active) > 1:
                active.sort(key=root_score, reverse=True)
                active = active[: max(1, math.ceil(len(active) / 2))]
        winner = max(active, key=root_score)

        # Improved root policy pi' = softmax(logits + sigma(q)) over ALL legal
        # actions (every child is leaf-evaluated, so no completion needed).
        # This is the policy-improvement output — logged as the future
        # distillation target for a learned prior.
        pi_prime = _softmax(logits + self._sigma(q_of(root), root))

        return decision_dict(
            root,
            action=winner,
            extras={
                "pi_prime": pi_prime,
                "prior_logits": np.asarray(logits, dtype=np.float64),
                "actions": [(b.mode, b.group) for b in root.branches],
            },
        )


class GumbelSwitcher(MPCSwitcher):
    """
    Drop-in switcher running Gumbel AlphaZero planning per real decision.

    Shares model construction and the ``decide`` contract with
    :class:`MPCSwitcher` (the inherited ``depth`` is unused).  The decision dict
    additionally carries ``pi_prime`` / ``prior_logits`` / ``actions`` so the
    eval harness can log policy-improvement targets.
    """

    def __init__(self, *args, prior=None, budget: int = 21, m: int = 4,
                 c_visit: float = 50.0, c_scale: float = 1.0,
                 gumbel_scale: float = 1.0, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.search = GumbelAlphaZero(
            prior=prior if prior is not None else HeuristicPrior(d_safe=self.d_safe),
            budget=budget, m=m, c_visit=c_visit, c_scale=c_scale,
            gumbel_scale=gumbel_scale, seed=seed,
        )

    def decide(self, robot_state: np.ndarray) -> dict:
        model = self._build_model(robot_state)
        ms = ForwardModel.state_from_robot_state(robot_state)
        decision = self.search.run(model, ms)
        self.decision_expansions.append(model.n_precise_expansions)
        return decision
