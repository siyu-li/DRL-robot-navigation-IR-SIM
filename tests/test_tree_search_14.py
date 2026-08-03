"""
Sim-free unit tests for the lazy Gumbel AlphaZero search (capswitcher_14).

A ``StubModel`` implements the single-edge transition API of ``ForwardModel14``
(``coarse_move`` / ``precise_next`` / ``n_transitions``) over tiny hand-written
or randomly generated MDPs.  The correctness anchor: with a deterministic root
(``gumbel_scale=0``) and a saturating budget, the search must reproduce the
exhaustive minimin value computed directly on the MDP spec (unsafe coarse
edges simply absent — pruning semantics, no collision penalty).
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE
from robot_nav.models.MARL.capswitcher_14.rl.search.common import (
    COLLISION_COST,
    QNormalizer,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import GumbelAlphaZero14
from robot_nav.models.MARL.capswitcher_14.rl.search.priors import UniformPrior
from robot_nav.models.MARL.capswitcher_14.rl.search.tree import (
    completed_q,
    expand_node,
    make_node,
    materialize,
)

# ---------------------------------------------------------------------------
# Stub model over abstract string states
# ---------------------------------------------------------------------------


class StubState:
    """Hashable stand-in for ModelState; ``poses`` feeds the search's seeding."""

    def __init__(self, name: str):
        self.name = name
        digest = abs(hash(name)) % (10 ** 6)
        self.poses = np.array([[float(digest), 0.0, 0.0]])

    def __repr__(self):
        return f"StubState({self.name})"


class _StubCandidate:
    def __init__(self, group: int, safe: bool):
        self.group = group
        self.safe = safe
        self.frames = [f"frames-{group}"] if safe else []
        self.clearance = 1.0 if safe else -1.0
        self.progress = 1.0


class _StubMove:
    def __init__(self, candidate, next_state):
        self.candidate = candidate
        self.next_state = next_state


class _StubCoarse:
    def __init__(self, n_groups: int):
        self._n = n_groups

    def selectable_groups(self) -> list[int]:
        return list(range(self._n))


class StubModel:
    """
    Single-edge-transition model over named states.

    Args:
        coarse_next:  {state: {group: next_state_name | None}} — ``None`` means
                      the shield refutes the group there; a missing group is
                      refuted too.
        precise_next_map: {state: next_state_name}.
        coarse_costs: per-group step cost (list, len n_groups).
        precise_cost: step cost of the precise edge.
        heuristic:    {state: leaf cost-to-go estimate}.
        terminals:    states where all robots have reached.
        collisions:   states predicted to be in collision (precise children).
    """

    def __init__(
        self,
        n_groups: int,
        coarse_next: dict,
        precise_next_map: dict,
        coarse_costs: list,
        precise_cost: float,
        heuristic: dict,
        terminals: set,
        collisions: set = frozenset(),
    ):
        self.coarse = _StubCoarse(n_groups)
        self._coarse_next = coarse_next
        self._precise_next = precise_next_map
        self._coarse_costs = list(coarse_costs)
        self._precise_cost = float(precise_cost)
        self._h = dict(heuristic)
        self._terminals = set(terminals)
        self._collisions = set(collisions)
        self._states = {name: StubState(name) for name in self._all_names()}
        self.n_coarse_vets = 0
        self.n_precise_expansions = 0

    def _all_names(self):
        names = set(self._coarse_next) | set(self._precise_next) | set(self._h)
        names |= self._terminals | self._collisions
        for row in self._coarse_next.values():
            names |= {v for v in row.values() if v is not None}
        names |= set(self._precise_next.values())
        return names

    def state(self, name: str) -> StubState:
        return self._states[name]

    @property
    def n_transitions(self) -> int:
        return self.n_coarse_vets + self.n_precise_expansions

    # --- API used by the search ---------------------------------------

    def all_reached(self, ms: StubState) -> bool:
        return ms.name in self._terminals

    def collision_pred(self, ms: StubState) -> bool:
        return ms.name in self._collisions

    def cost_to_go(self, ms: StubState) -> float:
        return float(self._h.get(ms.name, 0.0))

    def step_cost(
        self, mode: int, ms: StubState, group: int | None = None
    ) -> float:
        if mode == COARSE:
            return self._coarse_costs[group]
        return self._precise_cost

    def coarse_move(self, ms: StubState, group: int) -> _StubMove:
        self.n_coarse_vets += 1
        nxt = self._coarse_next.get(ms.name, {}).get(group)
        if nxt is None:
            return _StubMove(_StubCandidate(group, safe=False), None)
        return _StubMove(_StubCandidate(group, safe=True), self.state(nxt))

    def precise_next(self, ms: StubState) -> StubState:
        self.n_precise_expansions += 1
        return self.state(self._precise_next[ms.name])

    # --- Exhaustive reference (pruning semantics: unsafe edges absent) --

    def minimin(self, name: str, _seen: frozenset = frozenset()) -> float:
        if name in self._terminals:
            return 0.0
        if name in self._collisions:
            return COLLISION_COST
        assert name not in _seen, "reference recursion requires acyclic specs"
        seen = _seen | {name}
        options = []
        for g, nxt in self._coarse_next.get(name, {}).items():
            if nxt is not None:
                options.append(self._coarse_costs[g] + self.minimin(nxt, seen))
        options.append(self._precise_cost + self.minimin(self._precise_next[name], seen))
        return min(options)


# ---------------------------------------------------------------------------
# Hand-written MDPs
# ---------------------------------------------------------------------------


def small_mdp() -> tuple[StubModel, StubState]:
    """
    Two coarse groups + precise.  Group 0 is cheap and reaches the goal in two
    coarse hops; group 1 is safe but wasteful; precise is exact but expensive.
    Optimal plan: g0, g0 → cost 2.0.
    """
    model = StubModel(
        n_groups=2,
        coarse_next={
            "s0": {0: "s1", 1: "w1"},
            "s1": {0: "goal", 1: "w2"},
            "w1": {},
            "w2": {},
        },
        precise_next_map={"s0": "p1", "s1": "goal", "w1": "goal", "w2": "goal",
                          "p1": "goal"},
        coarse_costs=[1.0, 1.5],
        precise_cost=3.0,
        heuristic={"s0": 4.0, "s1": 2.5, "w1": 3.5, "w2": 3.5, "p1": 3.0},
        terminals={"goal"},
    )
    return model, model.state("s0")


def unsafe_below_root_mdp() -> tuple[StubModel, StubState]:
    """
    Group 0 is safe at the root but *refuted at s1* (discovered only on
    materialisation below the root).  The reference treats the refuted edge as
    absent — optimal: g0 then g1, cost 1.0 + 2.0 = 3.0.
    """
    model = StubModel(
        n_groups=2,
        coarse_next={
            "s0": {0: "s1", 1: "w1"},
            "s1": {0: None, 1: "goal"},   # g0 refuted at s1
            "w1": {},
        },
        precise_next_map={"s0": "pp", "s1": "pp", "w1": "goal", "pp": "goal"},
        coarse_costs=[1.0, 2.0],
        precise_cost=5.0,
        heuristic={"s0": 6.0, "s1": 2.0, "w1": 6.0, "pp": 5.0},
        terminals={"goal"},
    )
    return model, model.state("s0")


def liar_mdp() -> tuple[StubModel, StubState]:
    """
    The heuristic wildly overestimates s1 (pessimistic leaf), but the path
    through s1 reaches the goal cheaply — the terminal certificate must beat
    the lying estimate.  True optimum: g0, g0 → 0.5 + 0.5 = 1.0.
    """
    model = StubModel(
        n_groups=1,
        coarse_next={"s0": {0: "s1"}, "s1": {0: "goal"}},
        precise_next_map={"s0": "goal", "s1": "goal"},
        coarse_costs=[0.5],
        precise_cost=2.0,
        heuristic={"s0": 2.0, "s1": 100.0},  # liar: s1 is actually 0.5 from goal
        terminals={"goal"},
    )
    return model, model.state("s0")


def _run_deterministic(model, s0, budget=500, m=8, prior=None):
    search = GumbelAlphaZero14(
        prior or UniformPrior(), budget=budget, m=m, gumbel_scale=0.0
    )
    return search.run(model, s0)


# ---------------------------------------------------------------------------
# Equivalence with the exhaustive reference
# ---------------------------------------------------------------------------


def test_full_budget_matches_exhaustive_minimin():
    model, s0 = small_mdp()
    decision = _run_deterministic(model, s0)
    assert decision["value"] == pytest.approx(model.minimin("s0"))
    assert decision["mode"] == COARSE and decision["group"] == 0


def test_liar_heuristic_beaten_by_certificate():
    model, s0 = liar_mdp()
    decision = _run_deterministic(model, s0)
    assert decision["value"] == pytest.approx(1.0)
    assert decision["mode"] == COARSE


def _random_layered_mdp(rng: np.random.Generator, n_groups=3, n_layers=3, width=3):
    """
    Random acyclic layered MDP; some coarse edges randomly refuted.

    Heuristics are set to the *exact* cost-to-go — the regime where a
    saturating budget must reproduce the exhaustive minimin (mirrors the
    6-robot randomized test).  With arbitrarily wrong heuristics that
    guarantee does not hold below the root: the deterministic π′ selection is
    near-argmax, so a pessimistically-estimated subtree may never be corrected
    (inherent to the AlphaZero family; root arms are protected by Sequential
    Halving's forced visits).
    """
    layers = [[f"L{d}_{i}" for i in range(width)] for d in range(n_layers)]
    coarse_next, precise_next_map, heuristic = {}, {}, {}
    for d, layer in enumerate(layers):
        nxt_layer = layers[d + 1] if d + 1 < n_layers else None
        for name in layer:
            row = {}
            for g in range(n_groups):
                if rng.random() < 0.25:
                    row[g] = None                      # refuted
                elif nxt_layer is None:
                    row[g] = "goal"
                else:
                    row[g] = nxt_layer[rng.integers(width)]
            coarse_next[name] = row
            precise_next_map[name] = (
                "goal" if nxt_layer is None else nxt_layer[rng.integers(width)]
            )
            heuristic[name] = 0.0  # placeholder, replaced by exact V* below
    model = StubModel(
        n_groups=n_groups,
        coarse_next=coarse_next,
        precise_next_map=precise_next_map,
        coarse_costs=[float(rng.uniform(0.5, 2.0)) for _ in range(n_groups)],
        precise_cost=float(rng.uniform(2.0, 4.0)),
        heuristic=heuristic,
        terminals={"goal"},
    )
    for name in heuristic:
        model._h[name] = model.minimin(name)  # exact cost-to-go
    return model, model.state("L0_0")


def test_randomized_equivalence_with_minimin():
    rng = np.random.default_rng(7)
    for _ in range(10):
        model, s0 = _random_layered_mdp(rng)
        expected = model.minimin(s0.name)
        model2, s02 = model, s0
        # Fresh counters/model per run: rebuild an identical model.
        decision = _run_deterministic(model2, s02, budget=2000, m=4)
        assert decision["value"] == pytest.approx(expected), s0.name


# ---------------------------------------------------------------------------
# Lazy semantics: pruning, completion, budget
# ---------------------------------------------------------------------------


def test_refuted_edge_is_pruned_without_penalty():
    model, s0 = unsafe_below_root_mdp()
    decision = _run_deterministic(model, s0)
    # Reference treats the refuted edge as absent: optimum g0 → g1 = 3.0.
    assert decision["value"] == pytest.approx(3.0)
    assert decision["mode"] == COARSE and decision["group"] == 0
    # No COLLISION_COST leaked into the root value or pi_prime.
    assert decision["value"] < COLLISION_COST
    assert np.all(np.isfinite(decision["pi_prime"]))


def test_materialize_refuted_prunes_edge():
    model, s0 = unsafe_below_root_mdp()
    node = make_node(model, model.state("s1"))
    expand_node(node, model, UniformPrior())
    child = materialize(node, 0, model)          # g0 refuted at s1
    assert child is None
    assert not node.legal[0]
    assert node.children[0] is None
    assert node.N[0] == 0
    # The candidate is still stashed (label harvesting).
    assert node.branches[0].candidate is not None
    assert not node.branches[0].candidate.safe
    # Legal set never empty: precise survives.
    assert node.precise_action in node.legal_actions()


def test_completed_q_uses_vmix_for_unmaterialized_edges():
    model, s0 = small_mdp()
    node = make_node(model, s0)
    expand_node(node, model, UniformPrior())
    materialize(node, 0, model)                  # only g0 bought
    node.N[0] = 2                                 # pretend two visits
    qnorm = QNormalizer()
    cq = completed_q(node, qnorm)
    # Materialised edge: real normalised value; others: one shared v_mix.
    assert np.isfinite(cq[0]) and np.isfinite(cq[1]) and np.isfinite(cq[2])
    assert cq[1] == pytest.approx(cq[2])          # both unmaterialised → v_mix
    # v_mix interpolates node value and the visited edge's q: strictly between
    # (or equal at degenerate normalisation), never a collision-style 0 punish.
    lo, hi = sorted([qnorm.normalize(node.value()), cq[0]])
    assert lo - 1e-12 <= cq[1] <= hi + 1e-12


def test_budget_in_transitions_is_respected():
    for budget in (4, 6, 10, 30):
        model, s0 = small_mdp()
        GumbelAlphaZero14(UniformPrior(), budget=budget, gumbel_scale=0.0).run(
            model, s0
        )
        # Root eager verification (3 transitions here) always runs; beyond
        # that the budget is a hard cap.
        assert model.n_transitions <= max(budget, 3)


def test_root_candidates_cover_all_groups_with_clearances():
    model, s0 = unsafe_below_root_mdp()
    decision = _run_deterministic(model, s0)
    cands = decision["candidates"]
    assert len(cands) == 2                        # both groups vetted at root
    assert all(hasattr(c, "clearance") and hasattr(c, "safe") for c in cands)


def test_pi_prime_is_distribution_over_legal_preferring_cheap():
    model, s0 = small_mdp()
    decision = _run_deterministic(model, s0)
    pi = decision["pi_prime"]
    assert pi.shape == (3,)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= 0.0)
    # Cheapest optimal action dominates after a saturating search.
    assert int(np.argmax(pi)) == 0


def test_search_is_deterministic_given_seed():
    results = []
    for _ in range(2):
        model, s0 = small_mdp()
        search = GumbelAlphaZero14(UniformPrior(), budget=40, seed=3)
        results.append(search.run(model, s0))
    a, b = results
    assert a["mode"] == b["mode"] and a["group"] == b["group"]
    assert a["value"] == pytest.approx(b["value"])
    assert np.allclose(a["pi_prime"], b["pi_prime"])


def test_advisory_feasibility_mask_steers_but_never_overrides_vets():
    """A wrong 'infeasible' prediction may cost efficiency, never correctness;
    a wrong 'feasible' prediction is corrected by the vet on contact."""

    class MaskedUniform(UniformPrior):
        def feasibility(self, model, ms, branches):
            # Claims every coarse edge is infeasible (wrong for most).
            return np.array(
                [b.mode != COARSE for b in branches], dtype=bool
            )

    model, s0 = small_mdp()
    decision = _run_deterministic(model, s0, prior=MaskedUniform())
    # Root ignores predictions (exact vets), so the root decision is still
    # exact-optimal even under a fully wrong mask.
    assert decision["value"] == pytest.approx(model.minimin("s0"))


def test_transition_counter_charges_refuted_vets():
    model, _ = unsafe_below_root_mdp()
    node = make_node(model, model.state("s1"))
    expand_node(node, model, UniformPrior())
    before = model.n_transitions
    materialize(node, 0, model)                   # refuted
    assert model.n_transitions == before + 1
