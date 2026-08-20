"""
Sim-free unit tests for the plan-to-goal best-first baselines (A*, LevinTS,
PHS_h, PHS*) over the ``StubModel`` MDPs shared with the lazy-tree tests.

Anchors:

* the φ formulas match the paper (Orseau & Lelis 2021) by direct evaluation;
* A* with h ≡ 0 (Dijkstra) and with the *exact* cost-to-go reproduces the
  exhaustive minimin optimum;
* LevinTS under a uniform prior and unit losses expands in breadth-first
  order (the paper's degenerate case), and PHS_h / PHS* with h ≡ 0 collapse
  to LevinTS's expansion order;
* refuted vets are charged to the transition counters but never enter the
  plan; predicted-collision precise children are dead ends;
* the transition cap returns a generated goal plan when one exists, else the
  best partial path by g + h, flagged ``cap_hit``;
* the switcher wrapper replays a plan one decision per call, re-plans on
  exhaustion, and logs transitions only on planning decisions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE
from robot_nav.models.MARL.capswitcher_14.rl.search.best_first import (
    EVALUATIONS,
    BestFirstSearch14,
    PlanToGoalSwitcher14,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.priors import UniformPrior
from tests.test_tree_search_14 import (
    StubModel,
    _random_layered_mdp,
    liar_mdp,
    small_mdp,
    unsafe_below_root_mdp,
)


def _search(algo: str, **kw) -> BestFirstSearch14:
    return BestFirstSearch14(EVALUATIONS[algo], **kw)


def _zero_h(model: StubModel) -> StubModel:
    for k in model._h:
        model._h[k] = 0.0
    return model


def _exact_h(model: StubModel) -> StubModel:
    for k in model._h:
        model._h[k] = model.minimin(k)
    return model


# ---------------------------------------------------------------------------
# φ formulas (paper: A* f=g+h; LevinTS g/π; PHS_h (g+h)/π; PHS* (g+h)/π^(1+h/g))
# ---------------------------------------------------------------------------


def test_evaluation_formulas_match_paper():
    g, h, pi, depth = 2.0, 3.0, 0.25, 4
    lp = math.log(pi)
    assert EVALUATIONS["astar"](g, h, lp, depth) == pytest.approx(5.0)
    assert EVALUATIONS["levints"](g, h, lp, depth) == pytest.approx(
        math.log(g / pi)
    )
    assert EVALUATIONS["levints-depth"](g, h, lp, depth) == pytest.approx(
        math.log((depth + 1) / pi)
    )
    assert EVALUATIONS["phs"](g, h, lp, depth) == pytest.approx(
        math.log((g + h) / pi)
    )
    assert EVALUATIONS["phs-star"](g, h, lp, depth) == pytest.approx(
        math.log(g + h) - (1.0 + h / g) * lp
    )


def test_phs_star_reduces_to_phs_when_h_zero():
    g, pi = 1.5, 0.1
    lp = math.log(pi)
    assert EVALUATIONS["phs-star"](g, 0.0, lp, 2) == pytest.approx(
        EVALUATIONS["phs"](g, 0.0, lp, 2)
    )
    assert EVALUATIONS["phs"](g, 0.0, lp, 2) == pytest.approx(
        EVALUATIONS["levints"](g, 0.0, lp, 2)
    )


# ---------------------------------------------------------------------------
# Optimality anchors against the exhaustive minimin reference
# ---------------------------------------------------------------------------


def test_astar_zero_h_is_dijkstra_optimal():
    model, s0 = small_mdp()
    _zero_h(model)
    res = _search("astar").run(model, s0)
    assert res.solved and not res.cap_hit
    assert res.plan_cost == pytest.approx(model.minimin("s0"))
    assert [(d["mode"], d["group"]) for d in res.decisions] == [
        (COARSE, 0), (COARSE, 0)
    ]
    # Coarse decisions carry the vetted frames from materialisation.
    assert all(d["frames"] == ["frames-0"] for d in res.decisions)


def test_astar_exact_h_optimal_on_random_mdps():
    rng = np.random.default_rng(11)
    for _ in range(10):
        model, s0 = _random_layered_mdp(rng)   # heuristic already exact V*
        res = _search("astar").run(model, s0)
        assert res.solved
        assert res.plan_cost == pytest.approx(model.minimin(s0.name))


def test_liar_heuristic_still_solved():
    # h wildly overestimates s1; plan-to-goal search must still certify a
    # plan (first popped goal), though with inadmissible h it may be the
    # pricier route — only solvedness and in-model exactness are guaranteed.
    model, s0 = liar_mdp()
    res = _search("astar").run(model, s0)
    assert res.solved
    assert res.plan_cost < 1e9


def test_all_reached_root_returns_empty_solved_plan():
    model, _ = small_mdp()
    res = _search("astar").run(model, model.state("goal"))
    assert res.solved and res.decisions == [] and res.plan_cost == 0.0


# ---------------------------------------------------------------------------
# Expansion-order anchors (LevinTS / PHS family)
# ---------------------------------------------------------------------------


def _unit_cost_mdp() -> tuple[StubModel, object]:
    """Unit losses everywhere; deep enough to expose the expansion order."""
    model = StubModel(
        n_groups=2,
        coarse_next={
            "s0": {0: "a1", 1: "b1"},
            "a1": {0: "a2", 1: "b2"},
            "b1": {0: "b2", 1: "a2"},
            "a2": {0: "goal", 1: "goal"},
            "b2": {0: "goal", 1: "goal"},
        },
        precise_next_map={"s0": "a1", "a1": "a2", "b1": "b2", "a2": "goal",
                          "b2": "goal"},
        coarse_costs=[1.0, 1.0],
        precise_cost=1.0,
        heuristic={"s0": 0.0, "a1": 0.0, "b1": 0.0, "a2": 0.0, "b2": 0.0},
        terminals={"goal"},
    )
    return model, model.state("s0")


def test_levints_uniform_prior_expands_breadth_first():
    model, s0 = _unit_cost_mdp()
    trace: list = []
    res = _search("levints").run(model, s0, trace=trace)
    assert res.solved
    depths = [n.depth for n in trace]
    assert depths == sorted(depths)


def test_phs_family_with_zero_h_matches_levints_order():
    orders = {}
    for algo in ("levints", "phs", "phs-star"):
        model, s0 = _unit_cost_mdp()
        trace: list = []
        _search(algo).run(model, s0, trace=trace)
        orders[algo] = [n.ms.name for n in trace]
    assert orders["phs"] == orders["levints"]
    assert orders["phs-star"] == orders["levints"]


def test_levints_prefers_high_prior_probability_paths():
    class BiasedPrior:
        """Strongly favours group 1 everywhere."""

        def __call__(self, model, ms, branches):
            logits = np.zeros(len(branches))
            for i, b in enumerate(branches):
                if b.mode == COARSE and b.group == 1:
                    logits[i] = 10.0
            return logits

    model, s0 = _unit_cost_mdp()
    trace: list = []
    _search("levints", prior=BiasedPrior()).run(model, s0, trace=trace)
    # After the root, the first expanded node is the group-1 child.
    assert trace[1].branch.group == 1


# ---------------------------------------------------------------------------
# Domain semantics: refuted vets, dead ends, cap, accounting
# ---------------------------------------------------------------------------


def test_refuted_vets_charged_but_excluded_from_plan():
    model, s0 = unsafe_below_root_mdp()
    res = _search("astar").run(model, s0)
    assert res.solved
    # Reference treats the refuted edge as absent: optimum g0 -> g1 = 3.0.
    assert res.plan_cost == pytest.approx(3.0)
    assert [(d["mode"], d["group"]) for d in res.decisions] == [
        (COARSE, 0), (COARSE, 1)
    ]
    # The refuted vet at s1 was still charged to the counters the eval
    # harness reports.
    assert res.n_coarse_vets == model.n_coarse_vets
    assert res.n_coarse_vets >= 3    # root's two + at least the refuted one


def test_collision_precise_child_is_dead_end():
    model = StubModel(
        n_groups=1,
        coarse_next={"s0": {0: "s1"}, "s1": {0: "goal"}},
        precise_next_map={"s0": "boom", "s1": "boom"},
        coarse_costs=[1.0],
        precise_cost=0.1,            # tempting, but leads into a collision
        heuristic={"s0": 0.0, "s1": 0.0, "boom": 0.0},
        terminals={"goal"},
        collisions={"boom"},
    )
    res = _search("astar").run(model, model.state("s0"))
    assert res.solved
    assert all(d["mode"] == COARSE for d in res.decisions)
    assert res.plan_cost == pytest.approx(2.0)


def test_cap_returns_best_partial_with_flag():
    model, s0 = small_mdp()
    # Root expansion (2 vets + 1 precise = 3 transitions) always completes;
    # the cap then stops the search before any second expansion.
    res = _search("astar", max_transitions=2).run(model, s0)
    assert res.cap_hit and not res.solved
    # Best partial by g+h among the root's children: s1 (1.0 + 2.5).
    assert [(d["mode"], d["group"]) for d in res.decisions] == [(COARSE, 0)]
    assert res.plan_cost == pytest.approx(3.5)


def test_cap_still_returns_generated_goal_plan():
    # Cheap direct coarse edge to the goal: generated during the root
    # expansion, so even an immediate cap must return the certified plan.
    model = StubModel(
        n_groups=1,
        coarse_next={"s0": {0: "goal"}},
        precise_next_map={"s0": "p1", "p1": "goal"},
        coarse_costs=[1.0],
        # Cheaper f than the goal child, so p1 is popped first and the cap
        # fires before the goal node itself is ever popped.
        precise_cost=0.5,
        heuristic={"s0": 0.0, "p1": 0.0},
        terminals={"goal"},
    )
    res = _search("astar", max_transitions=1).run(model, model.state("s0"))
    assert res.solved and res.cap_hit
    assert res.plan_cost == pytest.approx(1.0)


def test_search_is_deterministic():
    runs = []
    for _ in range(2):
        model, s0 = small_mdp()
        res = _search("phs").run(model, s0)
        runs.append(([(d["mode"], d["group"]) for d in res.decisions],
                     res.plan_cost, res.expansions, res.n_coarse_vets))
    assert runs[0] == runs[1]


# ---------------------------------------------------------------------------
# Switcher wrapper: replay, re-plan, accounting
# ---------------------------------------------------------------------------


class _WrapperHarness(PlanToGoalSwitcher14):
    """Bypass the sim/backbone constructor plumbing and model building."""

    def __init__(self, model_factory, evaluate, **kw):
        self.search = BestFirstSearch14(evaluate, prior=kw.get("prior"))
        from collections import deque

        self._plan = deque()
        self.decision_transitions = []
        self.total_coarse_vets = 0
        self.total_precise_expansions = 0
        self.total_expansions = 0
        self.n_plans = 0
        self.n_solved_plans = 0
        self.n_cap_hits = 0
        self.n_fallbacks = 0
        self._model_factory = model_factory

    def _build(self, robot_state):
        return self._model_factory()


def test_wrapper_replays_plan_then_replans():
    def factory():
        model, s0 = small_mdp()
        _zero_h(model)
        return model, s0

    policy = _WrapperHarness(factory, EVALUATIONS["astar"])
    d1 = policy.decide(None)
    d2 = policy.decide(None)
    assert (d1["mode"], d1["group"]) == (COARSE, 0)
    assert (d2["mode"], d2["group"]) == (COARSE, 0)
    assert policy.n_plans == 1
    # Transitions logged on the planning decision only.
    assert policy.decision_transitions[0] > 0
    assert policy.decision_transitions[1] == 0
    # Plan exhausted -> the next decide() re-plans from the (fresh) state.
    policy.decide(None)
    assert policy.n_plans == 2
    assert policy.n_solved_plans == 2


def test_wrapper_reset_plan_drops_stale_suffix():
    def factory():
        model, s0 = small_mdp()
        return model, s0

    policy = _WrapperHarness(factory, EVALUATIONS["astar"])
    policy.decide(None)
    assert len(policy._plan) > 0
    policy.reset_plan()
    assert len(policy._plan) == 0
    policy.decide(None)
    assert policy.n_plans == 2


def test_wrapper_falls_back_to_precise_on_empty_plan():
    class RootOnlyModelFactory:
        """Cap so tight the plan is empty unless a goal child was generated."""

        def __call__(self):
            model = StubModel(
                n_groups=1,
                coarse_next={"s0": {0: "s1"}, "s1": {0: "goal"}},
                precise_next_map={"s0": "p1", "s1": "goal", "p1": "goal"},
                coarse_costs=[1.0],
                precise_cost=1.0,
                # Root is (falsely) the most promising node by g+h.
                heuristic={"s0": 0.0, "s1": 100.0, "p1": 100.0},
                terminals={"goal"},
            )
            return model, model.state("s0")

    policy = _WrapperHarness(RootOnlyModelFactory(), EVALUATIONS["astar"])
    policy.search.max_transitions = 1
    d = policy.decide(None)
    assert d["mode"] == PRECISE
    assert policy.n_fallbacks == 1
