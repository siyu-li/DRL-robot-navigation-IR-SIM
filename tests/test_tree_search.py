"""
Sim-free unit tests for the budgeted tree searches (MCTS / Gumbel AlphaZero).

A stub model over a small hand-built deterministic MDP replaces the analytic
``ForwardModel`` (same duck-typed interface the searches use), so everything
runs locally without irsim.  The exhaustive minimin ``plan_decision`` is the
correctness anchor: with enough budget the budgeted searches must reproduce it
exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE
from robot_nav.models.MARL.capswitcher.rl.shield import CoarseCandidate
from robot_nav.models.MARL.capswitcher.rl.forward_model import CoarseMove
from robot_nav.models.MARL.capswitcher.rl.search.minimin import plan_decision
from robot_nav.models.MARL.capswitcher.rl.search.common import (
    COLLISION_COST,
    Branch,
    QNormalizer,
)
from robot_nav.models.MARL.capswitcher.rl.search.mcts import MCTS
from robot_nav.models.MARL.capswitcher.rl.search.gumbel import GumbelAlphaZero
from robot_nav.models.MARL.capswitcher.rl.search.priors import (
    HeuristicPrior,
    UniformPrior,
)


class StubState:
    """Opaque MDP state with the ``poses`` array the seeding helpers expect."""

    _counter = 0

    def __init__(self, name: str) -> None:
        self.name = name
        StubState._counter += 1
        self.poses = np.full((1, 3), float(StubState._counter))

    def __repr__(self) -> str:  # pragma: no cover
        return f"StubState({self.name})"


class StubModel:
    """
    Deterministic finite MDP behind the ForwardModel interface used by the
    searches: coarse_moves / precise_next / step_cost / cost_to_go /
    collision_pred / all_reached, plus the expansion counter.

    ``coarse``  maps state-name -> {group: (next-name, safe)}.
    ``precise`` maps state-name -> next-name (always available).
    Step costs are state-independent (as in the real model): ``coarse_costs``
    per group, flat ``precise_cost``.
    """

    def __init__(self, coarse, precise, coarse_costs, precise_cost,
                 terminals, h, collisions=()) -> None:
        self._coarse = coarse
        self._precise = precise
        self.coarse_costs = coarse_costs
        self.precise_cost = precise_cost
        self.terminals = set(terminals)
        self.h = h
        self.collisions = set(collisions)
        self._states: dict[str, StubState] = {}
        self.n_precise_expansions = 0
        self.n_coarse_vets = 0

    def state(self, name: str) -> StubState:
        if name not in self._states:
            self._states[name] = StubState(name)
        return self._states[name]

    # --- ForwardModel interface -------------------------------------------

    def coarse_moves(self, ms):
        self.n_coarse_vets += 1
        moves = {}
        for group, (nxt, safe) in self._coarse.get(ms.name, {}).items():
            cand = CoarseCandidate(
                group=group, frames=[("frames", ms.name, group)],
                clearance=1.0 if safe else 0.0, progress=1.0, safe=safe,
            )
            moves[group] = CoarseMove(candidate=cand, next_state=self.state(nxt))
        return moves

    def precise_next(self, ms):
        self.n_precise_expansions += 1
        return self.state(self._precise[ms.name])

    def step_cost(self, action, ms, group=None):
        if action == COARSE:
            return self.coarse_costs[group]
        return self.precise_cost

    def cost_to_go(self, ms, alpha=None):
        return self.h.get(ms.name, 0.0)

    def collision_pred(self, ms):
        return ms.name in self.collisions

    def all_reached(self, ms):
        return ms.name in self.terminals


def small_mdp() -> tuple[StubModel, StubState]:
    """
    Root s0 with three options; unique optimum is c0 then precise (total 4.0):

        s0 --c0(1.0)--> sA --p(3.0)--> goalA          total 4.0   (optimal)
        s0 --c1(2.5)--> sB --c1(2.5)--> goalB         total 5.0
                        sB --p(3.0)--> sB1 --p--> goalB1  total 8.5
        s0 --p(3.0)---> sP --p(3.0)--> goalP          total 6.0
    """
    model = StubModel(
        coarse={
            "s0": {0: ("sA", True), 1: ("sB", True)},
            "sB": {1: ("goalB", True)},
        },
        precise={"s0": "sP", "sA": "goalA", "sB": "sB1",
                 "sB1": "goalB1", "sP": "goalP"},
        coarse_costs={0: 1.0, 1: 2.5},
        precise_cost=3.0,
        terminals={"goalA", "goalB", "goalB1", "goalP"},
        h={"sA": 3.5, "sB": 3.0, "sB1": 3.2, "sP": 3.1},
    )
    return model, model.state("s0")


def liar_mdp() -> tuple[StubModel, StubState]:
    """
    Certificate test: the heuristic lies that the precise branch is nearly free
    (h=0.1 forever down an endless chain), while c0 leads to a real terminal:

        s0 --c0(1.0)--> sA --p(3.0)--> goal        certified total 4.0
        s0 --p(3.0)---> lie --p--> lie --p--> ...  h always 0.1, never ends
    """
    model = StubModel(
        coarse={"s0": {0: ("sA", True)}},
        precise={"s0": "lie", "sA": "goal", "lie": "lie"},
        coarse_costs={0: 1.0},
        precise_cost=3.0,
        terminals={"goal"},
        h={"sA": 3.5, "lie": 0.1},
    )
    return model, model.state("s0")


# --------------------------------------------------------------------------
# Equivalence with exhaustive minimin
# --------------------------------------------------------------------------

def test_minimin_ground_truth():
    model, s0 = small_mdp()
    decision = plan_decision(model, s0, depth=3)
    assert decision["mode"] == COARSE and decision["group"] == 0
    assert decision["value"] == pytest.approx(4.0)


def test_mcts_matches_minimin_with_full_budget():
    for backup in ("bellman", "mean"):
        model, s0 = small_mdp()
        decision = MCTS(budget=50, backup=backup).run(model, s0)
        assert decision["mode"] == COARSE and decision["group"] == 0
        assert decision["value"] == pytest.approx(4.0)
        # Coarse decisions must return the exact vetted frames.
        assert decision["frames"] == [("frames", "s0", 0)]


def test_gumbel_matches_minimin_with_full_budget():
    model, s0 = small_mdp()
    search = GumbelAlphaZero(UniformPrior(), budget=50, gumbel_scale=0.0)
    decision = search.run(model, s0)
    assert decision["mode"] == COARSE and decision["group"] == 0
    assert decision["value"] == pytest.approx(4.0)


def _random_layered_mdp(
    rng: np.random.Generator, optimistic: bool = False
) -> tuple[StubModel, StubState]:
    """
    Random 3-layer DAG MDP (all paths terminal within 3 decisions) for
    randomized equivalence checks against exhaustive minimin.

    ``optimistic=True`` sets every leaf heuristic to 0 (an underestimate):
    unexpanded edges then always look attractive, so a value-guided search
    keeps expanding until estimates are exact — the regime where full-budget
    equivalence with exhaustive minimin is guaranteed.
    """
    layers = [[f"L{d}_{i}" for i in range(3)] for d in range(3)]
    terminals = {"goal"}
    coarse: dict = {}
    precise: dict = {}
    h: dict = {}
    for d, layer in enumerate(layers):
        nxt = layers[d + 1] if d + 1 < len(layers) else None
        for name in layer:
            h[name] = 0.0 if optimistic else float(rng.uniform(0.5, 6.0))
            pick = (lambda: str(rng.choice(nxt))) if nxt else (lambda: "goal")
            precise[name] = pick()
            groups = {}
            for g in (0, 1):
                if rng.random() < 0.7:
                    groups[g] = (pick(), bool(rng.random() < 0.8))
            if groups:
                coarse[name] = groups
    model = StubModel(
        coarse=coarse, precise=precise,
        coarse_costs={0: 1.3, 1: 2.1}, precise_cost=3.0,
        terminals=terminals, h=h,
    )
    return model, model.state("L0_0")


def test_randomized_equivalence_with_minimin():
    """
    Full-budget searches vs the exhaustive minimin value on random MDPs.

    MCTS: must match under arbitrary (noisy) leaf heuristics — UCT's
    exploration bonus eventually expands everything.

    Gumbel: must match under an *optimistic* (underestimating) heuristic.
    Under a pessimistic heuristic the deterministic non-root rule is
    exploitative by design (it trusts the leaf estimates and won't expand
    subtrees that look bad), so exhaustive equivalence is not guaranteed —
    that is inherent to the AlphaZero family, not a defect.
    """
    rng = np.random.default_rng(7)
    for _ in range(10):
        seed_state = rng.integers(1 << 30)
        model, s0 = _random_layered_mdp(np.random.default_rng(seed_state))
        expected = plan_decision(model, s0, depth=4)["value"]

        model2, s02 = _random_layered_mdp(np.random.default_rng(seed_state))
        got_mcts = MCTS(budget=500).run(model2, s02)["value"]
        assert got_mcts == pytest.approx(expected), f"MCTS != minimin (seed {seed_state})"

        model3, s03 = _random_layered_mdp(
            np.random.default_rng(seed_state), optimistic=True
        )
        expected_o = plan_decision(model3, s03, depth=4)["value"]
        model4, s04 = _random_layered_mdp(
            np.random.default_rng(seed_state), optimistic=True
        )
        got_gaz = GumbelAlphaZero(
            UniformPrior(), budget=500, gumbel_scale=0.0
        ).run(model4, s04)["value"]
        assert got_gaz == pytest.approx(expected_o), f"GAZ != minimin (seed {seed_state})"


# --------------------------------------------------------------------------
# Budget accounting
# --------------------------------------------------------------------------

def test_budget_is_respected():
    for budget in (1, 2, 3, 5):
        model, s0 = small_mdp()
        MCTS(budget=budget).run(model, s0)
        assert model.n_precise_expansions <= budget

        model, s0 = small_mdp()
        GumbelAlphaZero(UniformPrior(), budget=budget, gumbel_scale=0.0).run(model, s0)
        assert model.n_precise_expansions <= budget


def test_myopic_budget_still_decides():
    model, s0 = small_mdp()
    decision = MCTS(budget=1).run(model, s0)
    assert decision["mode"] in (COARSE, PRECISE)


# --------------------------------------------------------------------------
# Certificates vs estimates
# --------------------------------------------------------------------------

def test_certificate_overrides_lying_estimate():
    model, s0 = liar_mdp()
    search = MCTS(budget=12)
    # Access the root by re-running the internals: run() returns the decision;
    # certificates are visible through it (value = certified total).
    decision = search.run(model, s0)
    # The lie chain accumulates exact step costs (3.0 each) under Bellman
    # backup, so its estimate soon exceeds the certified 4.0 via c0.
    assert decision["mode"] == COARSE and decision["group"] == 0
    assert decision["value"] == pytest.approx(4.0)


def test_certificate_fields_are_separate():
    from robot_nav.models.MARL.capswitcher.rl.search.tree import (
        expand_node, make_node, simulate,
    )
    model, s0 = liar_mdp()
    root = make_node(model, s0)
    expand_node(root, model)
    # Descend the coarse edge to the terminal: certifies U along the path.
    coarse_idx = next(
        i for i, b in enumerate(root.branches) if b.mode == COARSE
    )
    lie_idx = next(i for i, b in enumerate(root.branches) if b.mode == PRECISE)
    for _ in range(2):
        simulate(root, model, lambda node, g: 0, first_action=coarse_idx)
    assert root.U == pytest.approx(4.0)              # certified incumbent
    assert root.children[coarse_idx].U == pytest.approx(3.0)
    assert root.children[lie_idx].U == float("inf")  # estimate only, no proof
    # The unexplored lie edge still *estimates* 3.0 + 0.1 < U, so the
    # effective value stays at the estimate (keep exploring, downside capped
    # by the certificate); only deeper search raises it past 4.0.
    assert root.q_hat == pytest.approx(3.1)
    assert root.value() == pytest.approx(3.1)


# --------------------------------------------------------------------------
# Collision handling
# --------------------------------------------------------------------------

def test_collision_child_is_dominated_and_never_expanded():
    model = StubModel(
        coarse={"s0": {0: ("sA", True)}},
        precise={"s0": "crash", "sA": "goal"},
        coarse_costs={0: 1.0},
        precise_cost=3.0,
        terminals={"goal"},
        h={"sA": 3.5},
        collisions={"crash"},
    )
    s0 = model.state("s0")
    decision = MCTS(budget=10).run(model, s0)
    assert decision["mode"] == COARSE and decision["group"] == 0
    # The collision state itself must never be expanded (only s0/sA are).
    assert model.n_precise_expansions <= 3
    minimin = plan_decision(model, s0, depth=2)
    assert minimin["mode"] == COARSE and minimin["group"] == 0


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_searches_are_deterministic():
    results = []
    for _ in range(2):
        model, s0 = small_mdp()
        results.append(MCTS(budget=7).run(model, s0))
    assert results[0]["mode"] == results[1]["mode"]
    assert results[0]["group"] == results[1]["group"]
    assert results[0]["value"] == pytest.approx(results[1]["value"])

    results = []
    for _ in range(2):
        model, s0 = small_mdp()
        search = GumbelAlphaZero(HeuristicPrior(), budget=7, seed=3)
        results.append(search.run(model, s0))
    assert results[0]["mode"] == results[1]["mode"]
    assert results[0]["group"] == results[1]["group"]
    np.testing.assert_allclose(results[0]["pi_prime"], results[1]["pi_prime"])


# --------------------------------------------------------------------------
# Gumbel policy-improvement output
# --------------------------------------------------------------------------

def test_gumbel_pi_prime_is_a_distribution_preferring_cheap_actions():
    model, s0 = small_mdp()
    search = GumbelAlphaZero(UniformPrior(), budget=50, gumbel_scale=0.0)
    decision = search.run(model, s0)
    pi = decision["pi_prime"]
    assert pi.shape == (len(decision["actions"]),)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= 0.0)
    # With a uniform prior, pi' must rank the optimal action (c0) highest.
    best = int(np.argmax(pi))
    assert decision["actions"][best] == (COARSE, 0)


# --------------------------------------------------------------------------
# Heuristic prior
# --------------------------------------------------------------------------

def _branch(mode, group, progress, clearance, step_cost):
    cand = None
    if mode == COARSE:
        cand = CoarseCandidate(group=group, frames=[], clearance=clearance,
                               progress=progress, safe=True)
    return Branch(mode=mode, group=group, frames=None, child=None,
                  step_cost=step_cost, candidate=cand)


def test_heuristic_prior_orders_by_efficiency():
    model, s0 = small_mdp()
    model.d_safe = 0.3
    branches = [
        _branch(COARSE, 0, progress=1.5, clearance=0.6, step_cost=1.5),  # ratio 1.0
        _branch(COARSE, 1, progress=0.3, clearance=0.6, step_cost=1.5),  # ratio 0.2
        _branch(PRECISE, None, progress=0.0, clearance=0.0, step_cost=11.7),
    ]
    prior = HeuristicPrior()
    logits = prior(model, s0, branches)
    assert logits.shape == (3,)
    assert np.all(np.isfinite(logits))
    assert logits[0] > logits[1]
    # Softmax of logits is a distribution.
    p = np.exp(logits - logits.max())
    assert (p / p.sum()).sum() == pytest.approx(1.0)


def test_heuristic_prior_precise_only():
    model, s0 = small_mdp()
    branches = [_branch(PRECISE, None, 0.0, 0.0, 11.7)]
    logits = HeuristicPrior(precise_bias=-1.0)(model, s0, branches)
    assert logits.shape == (1,) and np.isfinite(logits[0])


# --------------------------------------------------------------------------
# QNormalizer
# --------------------------------------------------------------------------

def test_qnormalizer():
    qn = QNormalizer()
    assert qn.normalize(5.0) == pytest.approx(0.5)   # no bounds yet
    qn.update(2.0)
    qn.update(10.0)
    qn.update(COLLISION_COST)                        # ignored
    assert qn.normalize(2.0) == pytest.approx(1.0)   # cheapest -> best
    assert qn.normalize(10.0) == pytest.approx(0.0)
    assert qn.normalize(6.0) == pytest.approx(0.5)
    assert qn.normalize(COLLISION_COST) == 0.0
