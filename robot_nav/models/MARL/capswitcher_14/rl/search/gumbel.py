"""
Gumbel AlphaZero switcher over the lazy tree — planning-only (Danihelka et
al., ICLR 2022), 14-robot instantiation.

Division of labour per real decision:

* **Root — eager.**  Every coarse group is vetted by the real shield (22
  single-group vets) and the precise edge rolled out, so the root's legal set
  is exact, the executed action always carries exactly vetted frames, and all
  22 clearance labels are harvested for the feasibility head's dataset.  This
  costs 23 transitions of the budget — charged honestly.
* **Below the root — lazy.**  Nodes expand to stubs + one prior forward pass;
  transitions are bought one per descent (``tree.simulate``).  Selection uses
  the deterministic rule ``argmax π'(a) − N(a)/(1+ΣN)`` with π' built from
  **completed** Q-values (v_mix for unmaterialised edges), the prior's
  *predicted* feasibility mask steering it away from likely-refuted coarse
  edges.  The real vet corrects a wrong prediction on contact by pruning.
* **Root bandit.**  Gumbel-top-m over the legal root edges, Sequential Halving
  by ``gum(a) + logits(a) + σ(q(a))``, act with the survivor.  ``σ(q) =
  (c_visit + max_b N(b)) · c_scale · q`` on min–max-normalised negated costs.
* **Distillation output.**  The improved root policy ``π' = softmax(logits +
  σ(completedQ))`` over legal edges is returned on every decision — the
  root-only distillation target for the learned prior (settled: interior
  nodes are not harvested).

Budget = ``model.n_transitions`` (single-group vets + precise rollouts),
including the root's 23 and any vet the shield refutes.

Known property (inherent to the AlphaZero family, sharpened by laziness): the
non-root rule is *exploitative* — with the paper's ``σ`` scaling, π' is
near-argmax, so a subtree whose leaf estimate is pessimistic (overestimates
cost) may never be materialised and its value never corrected downward.
Optimistic errors self-correct on contact; pessimistic ones persist.  Root
arms are protected by Sequential Halving's forced visits.  Full-budget
equivalence with exhaustive minimin therefore holds when leaf estimates are
exact (see the randomized test), not for arbitrary estimates.
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
from robot_nav.models.MARL.capswitcher_14.rl.search.priors import UniformPrior
from robot_nav.models.MARL.capswitcher_14.rl.search.tree import (
    Node,
    _softmax,
    bellman_refresh,
    completed_q,
    edge_value,
    expand_node,
    make_node,
    materialize,
    simulate,
)


def _state_seed(poses: np.ndarray, salt: int) -> int:
    """Stable 32-bit seed from rounded poses (reproducible planning per state)."""
    key = np.ascontiguousarray(np.round(poses, 6)).tobytes()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ (salt * 0x9E3779B1)) & 0xFFFFFFFF


class GumbelAlphaZero14:
    """
    One search instance (fresh per real decision, receding horizon).

    Args:
        prior:        Base policy ``prior(model, ms, branches) -> logits`` over
                      stubs; may expose ``feasibility(...) -> bool array``.
        budget:       Max model transitions (the root's eager 23 count).
        m:            Max root actions sampled without replacement by
                      Gumbel-top-m (clipped to the legal count).
        c_visit, c_scale: The ``σ`` transform constants (paper defaults).
        gumbel_scale: Scale of the root Gumbel noise; 0 = deterministic root
                      (equivalence tests).
        seed:         Mixed with a state hash per decision.
    """

    def __init__(
        self,
        prior,
        budget: int = 100,
        m: int = 16,
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

    def _pi_prime(self, node: Node, qnorm: QNormalizer) -> np.ndarray:
        """π' = softmax(logits + σ(completedQ)) over legal edges (0 elsewhere)."""
        cq = completed_q(node, qnorm)
        legal = node.legal_actions()
        scores = node.prior_logits[legal] + self._sigma(cq[legal], node)
        out = np.zeros(len(node.branches), dtype=np.float64)
        out[legal] = _softmax(scores)
        return out

    def run(self, model: ForwardModel14, ms) -> dict:
        root = make_node(model, ms)
        expand_node(root, model, self.prior)

        # Root: eager exact verification of every edge (the safety contract and
        # the label harvest).  Unsafe coarse edges are pruned here, before the
        # bandit ever sees them.
        for a in range(len(root.branches)):
            materialize(root, a, model)
        bellman_refresh(root)

        k = len(root.branches)
        logits = root.prior_logits
        legal = root.legal_actions()

        rng = np.random.default_rng(
            (_state_seed(ms.poses, 1) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        )
        gum = rng.gumbel(size=k) * self.gumbel_scale

        qnorm = QNormalizer()

        # Predicted-feasibility cache (advisory mask from the prior, if any).
        feas_fn = getattr(self.prior, "feasibility", None)
        feas_cache: dict[int, np.ndarray] = {}

        def predicted_feasible(node: Node) -> np.ndarray | None:
            if feas_fn is None:
                return None
            key = id(node)
            if key not in feas_cache:
                feas_cache[key] = np.asarray(
                    feas_fn(model, node.ms, node.branches), dtype=bool
                )
            return feas_cache[key]

        def select_action(node: Node, g: float) -> int:
            legal_n = node.legal_actions()
            allowed = [
                a for a in legal_n
                if g + node.branches[a].step_cost < root.U
            ] or legal_n
            feas = predicted_feasible(node)
            if feas is not None:
                # Advisory mask: applies only to unmaterialised coarse edges
                # (real vet results and the precise edge always stand).
                trusted = [
                    a for a in allowed
                    if node.children[a] is not None
                    or a == node.precise_action
                    or feas[a]
                ]
                allowed = trusted or allowed
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
        attempts, max_attempts = 0, 16 * max(self.budget, 1)
        for phase in range(phases):
            if model.n_transitions >= self.budget:
                break
            remaining = max(self.budget - model.n_transitions, 0)
            remaining_phases = phases - phase
            visits = max(1, remaining // (remaining_phases * len(active)))
            for a in active:
                for _ in range(visits):
                    if model.n_transitions >= self.budget or attempts >= max_attempts:
                        break
                    simulate(root, model, select_action, self.prior, first_action=a)
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
            # Distillation targets (root-only, settled).
            "pi_prime": pi_prime,
            "prior_logits": np.asarray(logits, dtype=np.float64),
            "legal": root.legal.copy(),
            "actions": [(br.mode, br.group) for br in root.branches],
            "n_transitions": model.n_transitions,
        }


class GumbelSwitcher14:
    """
    Drop-in switcher running lazy Gumbel AlphaZero planning per real decision.

    Same ``decide(robot_state) -> dict`` contract as the 6-robot switchers, so
    it plugs into ``SwitcherEnv.step(mode, group=..., frames=...)`` and the
    eval harness.  The decision dict additionally carries ``pi_prime`` /
    ``prior_logits`` / ``legal`` / ``actions`` / ``n_transitions`` for
    phase-0+ logging.

    Args mirror ``capswitcher``'s ``MPCSwitcher`` (minus ``depth``/``alpha``),
    plus the search hyper-parameters.
    """

    def __init__(
        self,
        backbone,
        coarse,
        sim,
        prior=None,
        budget: int = 100,
        m: int = 16,
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
                "GumbelSwitcher14 requires a SwitcherCost (load "
                "cost_14robots.yaml with SwitcherCost.from_yaml)"
            )
        self.backbone = backbone
        self.coarse = coarse
        self.sim = sim
        # Optional GroupFeatureBuilder: when set, every decision dict carries
        # the root's (group_feats, global_feats) — phase-0 shard logging.
        self.feature_builder = feature_builder
        self.d_safe = float(d_safe)
        self.selection_interval = int(selection_interval)
        self.goal_threshold = float(goal_threshold)
        self.cost = cost
        self.default_rho = float(default_rho)
        self.leaf_value = leaf_value
        self.search = GumbelAlphaZero14(
            prior=prior if prior is not None else UniformPrior(),
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
