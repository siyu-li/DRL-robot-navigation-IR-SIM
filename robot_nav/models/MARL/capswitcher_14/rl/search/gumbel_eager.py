"""
Gumbel AlphaZero switcher over the **eager** tree — GAZ14-E, the 14-robot
restatement of ``capswitcher.rl.search.gumbel`` (Danihelka et al., ICLR 2022,
planning-only variant).

Sibling of ``gumbel.py`` (GAZ14-L, lazy).  The two differ in exactly one
dimension — *what an expansion buys* — and are otherwise held identical
(σ transform, Sequential Halving, branch-and-bound mask, decision-dict
contract, budget unit) so a budget-matched comparison isolates that variable.

Division of labour per real decision:

* **Every node — eager.**  All 22 coarse groups are vetted by the real shield
  and the precise edge rolled out, at the root *and* at every interior node
  the search expands.  Consequently every legal edge carries a real Q and
  **no completed-Q / v_mix appears anywhere**: ``π′ = softmax(logits + σ(q))``
  is built on measured values.  This is affordable because the nonlinear
  rotation solve got cheap (``CoarseSteering14`` ``bfgs`` / ``bfgs_lean``).
* **The prior sees the vet.**  Ranking happens after materialisation, so
  :class:`priors_eager.HeuristicPrior14` scores edges on progress-per-cost and
  clearance margin — the dependence the lazy prior is forbidden.
* **Root bandit.**  Gumbel-top-m over the legal root edges, Sequential Halving
  by ``gum(a) + logits(a) + σ(q(a))``, act with the survivor.  ``σ(q) =
  (c_visit + max_b N(b)) · c_scale · q`` on min–max-normalised negated costs.
* **Distillation output.**  The improved root policy ``π′`` over legal edges is
  returned on every decision, in the same fixed 23-wide layout GAZ14-L logs —
  so both variants feed ``train_prior.py`` unchanged, and E's targets (built
  from exact Q) are a strictly better-informed teacher.

Budget
------
``model.n_transitions``, the same unit as GAZ14-L, so a budget number means the
same amount of model work in both.  One expansion costs
``tree_eager.expansion_cost`` = 23 transitions *exactly*, so a budget buys
``budget // 23`` nodes and the natural budget grid is multiples of 23:

    23  → root only = exhaustive depth-1 minimin  ("vet everything, act greedily")
    115 → root + 4 interior nodes
    253 → root + 10

**Expansions are atomic**: the search stops as soon as the remaining budget
cannot pay for a whole one, so a run never overshoots. The root expansion always
runs (it is the safety contract and the label harvest), so a budget below 23 is
rounded up to that single expansion — ``eval_gaz14_eager.py`` rejects it up
front rather than reporting a budget that was not honoured.

Known property, and how it differs from the lazy variant: below the root the
selection rule is *exploitative* (with the paper's σ scaling ``π′`` is
near-argmax), so a **node** whose leaf estimate is pessimistic may never be
expanded and its value never corrected downward.  Unlike GAZ14-L that risk does
not extend to individual *edges* — every edge of an expanded node is priced on a
real transition, never on a completion estimate.  Root arms are protected by
Sequential Halving's forced visits.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    ForwardModel14,
    build_forward_model,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.common import QNormalizer
from robot_nav.models.MARL.capswitcher_14.rl.search.priors_eager import (
    HeuristicPrior14,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.tree import (
    Node,
    _softmax,
    edge_value,
    make_node,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.tree_eager import (
    expand_node_eager,
    expansion_cost,
    simulate_eager,
)


def _state_seed(poses: np.ndarray, salt: int) -> int:
    """Stable 32-bit seed from rounded poses (reproducible planning per state)."""
    key = np.ascontiguousarray(np.round(poses, 6)).tobytes()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ (salt * 0x9E3779B1)) & 0xFFFFFFFF


class GumbelAlphaZero14Eager:
    """
    One search instance (fresh per real decision, receding horizon).

    Args:
        prior:        Base policy ``prior(model, ms, branches) -> logits``,
                      called **after** materialisation so it may read the vet
                      (see ``priors_eager.py``).  Any ``feasibility`` attribute
                      is ignored — this variant vets for real everywhere.
        budget:       Max model transitions.  Effective search size is
                      ``budget // 23`` node expansions; the root always runs.
        m:            Max root actions sampled without replacement by
                      Gumbel-top-m (clipped to the legal count).  Keep it below
                      ``(budget − 23) // 23`` or Sequential Halving has nothing
                      to allocate.
        c_visit, c_scale: The ``σ`` transform constants (paper defaults).
        gumbel_scale: Scale of the root Gumbel noise; 0 = deterministic root
                      (equivalence tests).
        seed:         Mixed with a state hash per decision.
    """

    def __init__(
        self,
        prior,
        budget: int = 115,
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

    def _sigma(self, q: np.ndarray, node: Node) -> np.ndarray:
        return (self.c_visit + float(node.N.max())) * self.c_scale * q

    def _q(self, node: Node, qnorm: QNormalizer) -> np.ndarray:
        """
        Normalised (higher-is-better) q per edge — real values throughout.

        Every legal edge of an eagerly expanded node is materialised, so there
        is nothing to complete; illegal (shield-refuted) edges get ``-inf`` so
        no selection rule can reach them.
        """
        out = np.full(len(node.branches), -np.inf, dtype=np.float64)
        legal = node.legal_actions()
        raw = {a: edge_value(node, a) for a in legal}
        for v in raw.values():
            qnorm.update(v)
        for a in legal:
            out[a] = qnorm.normalize(raw[a])
        return out

    def _pi_prime(self, node: Node, qnorm: QNormalizer) -> np.ndarray:
        """π' = softmax(logits + σ(q)) over legal edges (0 elsewhere)."""
        q = self._q(node, qnorm)
        legal = node.legal_actions()
        scores = node.prior_logits[legal] + self._sigma(q[legal], node)
        out = np.zeros(len(node.branches), dtype=np.float64)
        out[legal] = _softmax(scores)
        return out

    def run(self, model: ForwardModel14, ms) -> dict:
        root = make_node(model, ms)
        # Root expansion: the exact legal set, exactly vetted frames for the
        # executed action, and all 22 clearance labels for the feasibility head.
        expand_node_eager(root, model, self.prior)

        ecost = expansion_cost(model)
        logits = root.prior_logits
        legal = root.legal_actions()

        rng = np.random.default_rng(
            (_state_seed(ms.poses, 1) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        )
        gum = rng.gumbel(size=len(root.branches)) * self.gumbel_scale

        qnorm = QNormalizer()

        def select_action(node: Node, g: float) -> int:
            legal_n = node.legal_actions()
            # Branch-and-bound: positive step costs mean an edge whose exact
            # accumulated path cost already reaches the root's certificate can
            # never improve on it.
            allowed = [
                a for a in legal_n
                if g + node.branches[a].step_cost < root.U
            ] or legal_n
            pi = self._pi_prime(node, qnorm)
            total = float(node.N.sum())
            gap = pi - node.N / (1.0 + total)
            return max(allowed, key=lambda a: gap[a])

        def root_score(a: int) -> float:
            v = edge_value(root, a)
            qnorm.update(v)
            sig = self._sigma(np.array([qnorm.normalize(v)]), root)[0]
            return gum[a] + logits[a] + sig

        # ---- Gumbel-top-m + Sequential Halving over legal root edges -------
        m = min(self.m, len(legal))
        active = sorted(legal, key=lambda a: gum[a] + logits[a], reverse=True)[:m]
        phases = max(1, math.ceil(math.log2(m)) if m > 1 else 1)
        # Descents onto terminal / predicted-collision nodes expand nothing and
        # buy nothing; the cap keeps such an arm from soaking up a phase.
        attempts, max_attempts = 0, 16 * max(self.budget // max(ecost, 1), 1)
        for phase in range(phases):
            if model.n_transitions + ecost > self.budget:
                break
            affordable = (self.budget - model.n_transitions) // ecost
            remaining_phases = phases - phase
            visits = max(1, affordable // (remaining_phases * len(active)))
            for a in active:
                for _ in range(visits):
                    if (
                        model.n_transitions + ecost > self.budget
                        or attempts >= max_attempts
                    ):
                        break
                    simulate_eager(
                        root, model, select_action, self.prior, first_action=a
                    )
                    attempts += 1
            if len(active) > 1:
                active.sort(key=root_score, reverse=True)
                active = active[: max(1, math.ceil(len(active) / 2))]
        winner = max(active, key=root_score)

        pi_prime = self._pi_prime(root, qnorm)

        b = root.branches[winner]
        return {
            "mode": b.mode,
            "group": b.group,
            "frames": b.frames,
            # All 22 coarse candidates (safe and refuted) — clearance labels.
            "candidates": [
                br.candidate for br in root.branches if br.candidate is not None
            ],
            "value": float(edge_value(root, winner)),
            # Distillation targets (root-only, same layout as GAZ14-L).
            "pi_prime": pi_prime,
            "prior_logits": np.asarray(logits, dtype=np.float64),
            "legal": root.legal.copy(),
            "actions": [(br.mode, br.group) for br in root.branches],
            "n_transitions": model.n_transitions,
            "n_expansions": model.n_transitions // max(ecost, 1),
        }


class GumbelSwitcher14Eager:
    """
    Drop-in switcher running eager Gumbel AlphaZero planning per real decision.

    Same ``decide(robot_state) -> dict`` contract as :class:`GumbelSwitcher14`,
    so it plugs into ``SwitcherEnv.step(mode, group=..., frames=...)`` and
    reuses ``eval_gaz14_lazy``'s harness (``run`` / ``_gumbel_decider`` /
    ``_save_pi_targets``) verbatim.  The decision dict additionally carries
    ``n_expansions``.

    Args mirror :class:`GumbelSwitcher14` exactly, except that ``prior``
    defaults to :class:`HeuristicPrior14` rather than the uniform prior — the
    eager expansion has already paid for the statistics it reads, so there is
    no reason to start flat.
    """

    def __init__(
        self,
        backbone,
        coarse,
        sim,
        prior=None,
        budget: int = 115,
        m: int = 4,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        gumbel_scale: float = 1.0,
        seed: int = 0,
        d_safe: float = 0.3,
        selection_interval: int = 5,
        goal_threshold: float = 0.3,
        cost: SwitcherCost | None = None,
        default_rho: float = 0.2,
        leaf_value=None,
        feature_builder=None,
    ) -> None:
        if cost is None:
            raise ValueError(
                "GumbelSwitcher14Eager requires a SwitcherCost (load "
                "cost_14robots.yaml with SwitcherCost.from_yaml)"
            )
        self.backbone = backbone
        self.coarse = coarse
        self.sim = sim
        # Optional GroupFeatureBuilder: when set, every decision dict carries
        # the root's (group_feats, global_feats) — shard logging, shared with
        # GAZ14-L so both variants' shards are interchangeable.
        self.feature_builder = feature_builder
        self.d_safe = float(d_safe)
        self.selection_interval = int(selection_interval)
        self.goal_threshold = float(goal_threshold)
        self.cost = cost
        self.default_rho = float(default_rho)
        self.leaf_value = leaf_value
        self.search = GumbelAlphaZero14Eager(
            prior=prior if prior is not None else HeuristicPrior14(d_safe=d_safe),
            budget=budget, m=m, c_visit=c_visit, c_scale=c_scale,
            gumbel_scale=gumbel_scale, seed=seed,
        )
        # Per-decision transition counts (budget accounting for eval).
        self.decision_transitions: list[int] = []

    def _build_model(self, robot_state: np.ndarray) -> ForwardModel14:
        return build_forward_model(
            self.backbone,
            self.coarse,
            self.sim,
            robot_state,
            d_safe=self.d_safe,
            selection_interval=self.selection_interval,
            goal_threshold=self.goal_threshold,
            cost=self.cost,
            default_rho=self.default_rho,
            leaf_value=self.leaf_value,
        )

    def decide(self, robot_state: np.ndarray) -> dict:
        model = self._build_model(robot_state)
        ms = ForwardModel14.state_from_robot_state(robot_state)
        decision = self.search.run(model, ms)
        self.decision_transitions.append(model.n_transitions)
        if self.feature_builder is not None:
            gf, glf = self.feature_builder(model, ms)
            decision["group_feats"] = gf
            decision["global_feats"] = glf
        return decision
