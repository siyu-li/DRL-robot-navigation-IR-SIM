"""
Sim-free unit tests for the **eager** Gumbel AlphaZero search (GAZ14-E).

Reuses the ``StubModel`` MDPs of ``test_tree_search_14.py`` (the lazy variant's
suite) so both searches are held to the same correctness anchor: with a
deterministic root (``gumbel_scale=0``) and a saturating budget, the search
must reproduce the exhaustive minimin computed directly on the MDP spec, with
shield-refuted coarse edges simply absent.

The eager-specific properties pinned here:

* an expansion buys *every* edge and costs exactly ``expansion_cost``;
* expansions are **atomic** — spend is always a whole multiple of that cost, so
  a budget is never overshot by a half-bought node;
* every legal edge of an expanded node has a real child, so nothing anywhere is
  priced by a completion estimate (the lazy tree's ``v_mix``);
* :class:`HeuristicPrior14` ranks on the sunk vetting statistics, and its
  MAD standardisation is invariant to a global rescaling of the cost table.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE
from robot_nav.models.MARL.capswitcher_14.rl.search.common import (
    COLLISION_COST,
    Branch,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel_eager import (
    GumbelAlphaZero14Eager,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.priors_eager import (
    HeuristicPrior14,
    UniformPrior,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.tree import make_node
from robot_nav.models.MARL.capswitcher_14.rl.search.tree_eager import (
    expand_node_eager,
    expansion_cost,
)
from tests.test_tree_search_14 import (
    _random_layered_mdp,
    liar_mdp,
    small_mdp,
    unsafe_below_root_mdp,
)


def _run_deterministic(model, s0, budget=500, m=8, prior=None):
    search = GumbelAlphaZero14Eager(
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


def test_randomized_equivalence_with_minimin():
    rng = np.random.default_rng(7)
    for _ in range(10):
        model, s0 = _random_layered_mdp(rng)
        expected = model.minimin(s0.name)
        decision = _run_deterministic(model, s0, budget=2000, m=4)
        assert decision["value"] == pytest.approx(expected), s0.name


def test_refuted_edge_is_pruned_without_penalty():
    model, s0 = unsafe_below_root_mdp()
    decision = _run_deterministic(model, s0)
    # Reference treats the refuted edge as absent: optimum g0 → g1 = 3.0.
    assert decision["value"] == pytest.approx(3.0)
    assert decision["mode"] == COARSE and decision["group"] == 0
    assert decision["value"] < COLLISION_COST
    assert np.all(np.isfinite(decision["pi_prime"]))


# ---------------------------------------------------------------------------
# Eager semantics: whole-node expansion, atomic budget, no completion
# ---------------------------------------------------------------------------


def test_expansion_buys_every_edge_at_its_exact_cost():
    model, s0 = small_mdp()
    ecost = expansion_cost(model)
    assert ecost == 3                       # 2 coarse groups + precise

    node = make_node(model, s0)
    before = model.n_transitions
    expand_node_eager(node, model, UniformPrior())
    assert model.n_transitions - before == ecost
    # Every legal edge is materialised: nothing is left to a completion estimate.
    for a in node.legal_actions():
        assert node.children[a] is not None
    assert node.precise_action in node.legal_actions()


def test_refuted_coarse_edge_is_illegal_with_candidate_kept():
    model, _ = unsafe_below_root_mdp()
    node = make_node(model, model.state("s1"))
    expand_node_eager(node, model, UniformPrior())      # g0 is refuted at s1
    assert not node.legal[0]
    assert node.children[0] is None
    assert node.N[0] == 0
    # Candidate still stashed (root label harvesting), and the refuted vet was
    # charged like any other.
    assert node.branches[0].candidate is not None
    assert not node.branches[0].candidate.safe
    assert model.n_transitions == expansion_cost(model)


def test_prior_runs_after_vetting_and_sees_candidates():
    """The eager contract: branches carry vet output when the prior is called."""
    seen: list = []

    class _Recorder(UniformPrior):
        def __call__(self, model, ms, branches):
            seen.append(
                [(b.mode, b.candidate is not None, b.progress) for b in branches]
            )
            return super().__call__(model, ms, branches)

    model, s0 = small_mdp()
    node = make_node(model, s0)
    expand_node_eager(node, model, _Recorder())

    assert len(seen) == 1
    rows = seen[0]
    coarse_rows = [r for r in rows if r[0] == COARSE]
    assert coarse_rows and all(has_cand for _, has_cand, _ in coarse_rows)
    assert all(prog is not None for _, _, prog in coarse_rows)
    # StubModel exposes no goal_distances, so precise progress is unknown —
    # the prior must tolerate it rather than assume it.
    assert [r for r in rows if r[0] == PRECISE][0][2] is None


def test_budget_is_atomic_and_never_overshot():
    ecost = expansion_cost(small_mdp()[0])
    for budget in (3, 5, 6, 10, 30, 31):
        model, s0 = small_mdp()
        GumbelAlphaZero14Eager(
            UniformPrior(), budget=budget, gumbel_scale=0.0
        ).run(model, s0)
        # Whole expansions only, and (past the mandatory root) a hard cap.
        assert model.n_transitions % ecost == 0
        assert model.n_transitions <= max(budget, ecost)


def test_root_expansion_runs_even_below_one_expansion_budget():
    model, s0 = small_mdp()
    GumbelAlphaZero14Eager(UniformPrior(), budget=1, gumbel_scale=0.0).run(
        model, s0
    )
    # The root's vet is the safety contract — always paid, and reported.
    assert model.n_transitions == expansion_cost(model)


def test_decision_reports_expansions_and_transitions():
    model, s0 = small_mdp()
    decision = _run_deterministic(model, s0, budget=12)
    ecost = expansion_cost(model)
    assert decision["n_transitions"] == model.n_transitions
    assert decision["n_expansions"] == model.n_transitions // ecost


def test_root_candidates_cover_all_groups_with_clearances():
    model, s0 = unsafe_below_root_mdp()
    decision = _run_deterministic(model, s0)
    cands = decision["candidates"]
    assert len(cands) == 2
    assert all(hasattr(c, "clearance") and hasattr(c, "safe") for c in cands)


def test_pi_prime_is_distribution_over_legal_preferring_cheap():
    model, s0 = small_mdp()
    decision = _run_deterministic(model, s0)
    pi = decision["pi_prime"]
    assert pi.shape == (3,)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= 0.0)
    assert int(np.argmax(pi)) == 0


def test_pi_prime_puts_no_mass_on_refuted_edges():
    model, _ = unsafe_below_root_mdp()
    search = GumbelAlphaZero14Eager(UniformPrior(), budget=200, gumbel_scale=0.0)
    decision = search.run(model, model.state("s1"))    # g0 refuted here
    assert not decision["legal"][0]
    assert decision["pi_prime"][0] == 0.0
    assert decision["pi_prime"].sum() == pytest.approx(1.0)


def test_search_is_deterministic_given_seed():
    results = []
    for _ in range(2):
        model, s0 = small_mdp()
        search = GumbelAlphaZero14Eager(UniformPrior(), budget=40, seed=3)
        results.append(search.run(model, s0))
    a, b = results
    assert a["mode"] == b["mode"] and a["group"] == b["group"]
    assert a["value"] == pytest.approx(b["value"])
    assert np.allclose(a["pi_prime"], b["pi_prime"])


# ---------------------------------------------------------------------------
# HeuristicPrior14
# ---------------------------------------------------------------------------


class _Cand:
    def __init__(self, group, clearance, progress, safe=True):
        self.group = group
        self.clearance = float(clearance)
        self.progress = float(progress)
        self.safe = bool(safe)
        self.frames = []


class _PriorModel:
    """Minimal stand-in exposing only what the prior reads."""

    def __init__(self, d_safe=0.3):
        self.d_safe = float(d_safe)


def _coarse_branch(group, step_cost, progress, clearance, safe=True):
    cand = _Cand(group, clearance, progress, safe)
    return Branch(
        mode=COARSE, group=group, step_cost=float(step_cost),
        candidate=cand, progress=float(progress),
    )


def _precise_branch(step_cost, progress=None):
    return Branch(
        mode=PRECISE, group=None, step_cost=float(step_cost), progress=progress,
    )


def _logits(branches, prior=None, d_safe=0.3):
    prior = prior or HeuristicPrior14()
    return prior(_PriorModel(d_safe), object(), branches)


def test_heuristic_prior_prefers_higher_progress_per_cost():
    branches = [
        _coarse_branch(0, step_cost=10.0, progress=1.0, clearance=1.0),
        _coarse_branch(1, step_cost=10.0, progress=5.0, clearance=1.0),
        _coarse_branch(2, step_cost=10.0, progress=3.0, clearance=1.0),
        _precise_branch(100.0, progress=1.0),
    ]
    lg = _logits(branches)
    assert lg[1] > lg[2] > lg[0]


def test_heuristic_prior_breaks_ties_on_clearance():
    branches = [
        _coarse_branch(0, step_cost=10.0, progress=2.0, clearance=0.35),
        _coarse_branch(1, step_cost=10.0, progress=2.0, clearance=1.5),
        _precise_branch(100.0, progress=1.0),
    ]
    lg = _logits(branches)
    assert lg[1] > lg[0]


def test_heuristic_prior_gives_refuted_edges_no_score():
    branches = [
        _coarse_branch(0, step_cost=10.0, progress=99.0, clearance=5.0, safe=False),
        _coarse_branch(1, step_cost=10.0, progress=1.0, clearance=1.0),
        _coarse_branch(2, step_cost=10.0, progress=3.0, clearance=1.0),
        _precise_branch(100.0, progress=1.0),
    ]
    lg = _logits(branches)
    # The refuted edge is inert (masked by `legal` upstream) and — crucially —
    # its huge progress does not distort the scale the legal edges share.
    assert lg[0] == 0.0
    legal_only = _logits(branches[1:])
    assert np.allclose(lg[1:], legal_only)


def test_heuristic_prior_is_invariant_to_cost_table_rescaling():
    """MAD standardisation ⇒ retuning prices does not silently retune the prior."""
    base = [
        _coarse_branch(0, step_cost=52.7, progress=1.2, clearance=0.9),
        _coarse_branch(1, step_cost=52.7, progress=2.4, clearance=0.5),
        _coarse_branch(2, step_cost=8.8, progress=6.0, clearance=1.4),
        _precise_branch(810.0, progress=9.0),
    ]
    scaled = [
        _coarse_branch(0, step_cost=527.0, progress=1.2, clearance=0.9),
        _coarse_branch(1, step_cost=527.0, progress=2.4, clearance=0.5),
        _coarse_branch(2, step_cost=88.0, progress=6.0, clearance=1.4),
        _precise_branch(8100.0, progress=9.0),
    ]
    assert np.allclose(_logits(base), _logits(scaled))


def test_heuristic_prior_resolves_within_class_despite_outliers():
    """
    The real cost table's four size-7 groups (cost 8.8) are extreme ``eff``
    outliers next to the eighteen size-3/4 groups (cost 52.7).  A plain z-score
    would squash the eighteen together; the MAD scale must keep them apart
    while the outliers saturate.
    """
    branches = [
        _coarse_branch(g, step_cost=52.7, progress=1.0 + 0.25 * g, clearance=1.0)
        for g in range(18)
    ]
    branches += [
        _coarse_branch(18 + g, step_cost=8.8, progress=3.0 + g, clearance=1.0)
        for g in range(4)
    ]
    branches.append(_precise_branch(810.0, progress=9.0))

    prior = HeuristicPrior14(z_clip=3.0, w_eff=1.0)
    lg = _logits(branches, prior)
    within, outliers = lg[:18], lg[18:22]

    # Strictly increasing across the size-3/4 class — not collapsed.
    assert np.all(np.diff(within) > 1e-3)
    # The cheap size-7 groups dominate, stay bounded by the saturation level...
    assert outliers.min() > within.max()
    assert np.all(outliers < 3.0 + 1e-9)
    # ...and — the reason saturation is tanh rather than a hard clip — remain
    # *ordered* by efficiency out in the tail instead of pinning to one value.
    assert np.all(np.diff(outliers) > 1e-6)


def test_heuristic_prior_saturation_is_strictly_monotone():
    """
    No two distinct efficiencies may collapse onto the same score, and the
    score stays bounded by ``z_clip``.

    Monotonicity is exact only while ``tanh`` has float64 resolution left —
    past roughly 19·z_clip MAD units it returns exactly 1.0 and ordering does
    saturate.  The real table spreads about 10 MAD units, so the range probed
    here (0 → 50×) sits far inside that.
    """
    prior = HeuristicPrior14(z_clip=3.0, w_clear=0.0)
    branches = [
        _coarse_branch(g, step_cost=1.0, progress=p, clearance=1.0)
        for g, p in enumerate([0.0, 1.0, 2.0, 5.0, 20.0, 50.0])
    ]
    lg = _logits(branches, prior)
    assert np.all(np.diff(lg) > 0.0)
    assert lg.max() <= 3.0


def test_heuristic_prior_scores_precise_on_measured_progress():
    """Precise competes on progress-per-cost, not on a hand-set bias."""
    cheap = [
        _coarse_branch(0, step_cost=10.0, progress=1.0, clearance=1.0),
        _coarse_branch(1, step_cost=10.0, progress=1.2, clearance=1.0),
        _precise_branch(10.0, progress=8.0),      # very efficient precise
    ]
    dear = [
        _coarse_branch(0, step_cost=10.0, progress=1.0, clearance=1.0),
        _coarse_branch(1, step_cost=10.0, progress=1.2, clearance=1.0),
        _precise_branch(800.0, progress=0.1),     # very inefficient precise
    ]
    assert _logits(cheap)[2] > _logits(dear)[2]


def test_heuristic_prior_tolerates_unknown_precise_progress():
    branches = [
        _coarse_branch(0, step_cost=10.0, progress=1.0, clearance=1.0),
        _coarse_branch(1, step_cost=10.0, progress=3.0, clearance=1.0),
        _precise_branch(100.0, progress=None),
    ]
    lg = _logits(branches)
    assert np.all(np.isfinite(lg))
    assert lg[2] == pytest.approx(0.0)          # unknown → treated as median


def test_heuristic_prior_end_to_end_matches_optimum_on_stub_mdp():
    """A non-flat prior must not break the saturating-budget correctness anchor."""
    model, s0 = small_mdp()
    decision = _run_deterministic(model, s0, prior=HeuristicPrior14())
    assert decision["value"] == pytest.approx(model.minimin("s0"))
