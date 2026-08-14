"""
Wall-clock profile of the ``iterate_gaz14`` plan→distil loop.

Answers, with measured numbers rather than guesses:

* how long **materialising the precise edge** takes (the GAT rollout:
  ``selection_interval × n_unreached`` frozen-actor forwards per edge), split
  into the root's eager one and the ones bought inside the tree;
* how long **materialising the coarse edges** takes (the root vets every one of
  the 22 groups eagerly — that is the "materialize all coarse" cost — plus the
  vets bought during descents);
* how long the **simulator** takes (``SwitcherEnv.step``: irsim sub-steps plus
  the env's own GAT forward for precise actions);
* how long **distillation** takes (``train_prior`` per cycle).

Nothing in the search source is modified: timers are monkey-patched in for the
run and removed afterwards (see :mod:`robot_nav.profiling`).

Usage (GPU box — the sim does not run locally):

    # same flags as iterate_gaz14, plus profiler flags before/after them
    python -m robot_nav.profile_gaz14 --iterations 1 --episodes 3 \
        --budget 100 --out runs/prof_gaz14 --report-json runs/prof_gaz14/profile.json

    # learned leaf / learned prior path
    python -m robot_nav.profile_gaz14 --iterations 2 --episodes 5 --budget 100 \
        --value-model <ckpt> --out runs/prof_gaz14_value

A few episodes are enough: every number below is a per-call mean, and a
decision costs the same whether it is the 3rd or the 300th.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from robot_nav.profiling import Profiler, Stat

# ---- labels (also the report's row order) ---------------------------------

L_EVAL         = "EVAL run (all episodes: plan + sim)"
L_DECIDE       = "PLAN decision (total)"
L_BUILD_MODEL  = "build forward model"
L_SEARCH       = "gumbel search.run"
L_ROOT_COARSE  = "materialize ROOT coarse vet"
L_ROOT_PRECISE = "materialize ROOT precise rollout"
L_TREE_COARSE  = "materialize tree coarse vet"
L_TREE_PRECISE = "materialize tree precise rollout"
L_SIMULATE     = "tree descent (simulate)"
L_EXPAND       = "expand node (stubs + prior)"
L_PRIOR        = "prior net forward"
L_FEATS        = "group features"
L_MAKE_NODE    = "leaf node eval"
L_COST_TO_GO   = "leaf cost-to-go"
L_MODEL_RS     = "model.robot_state (GAT state prep)"
L_COARSE_ACT   = "coarse.compute_actions"
# --- obstacle / collision checking, at each of the three places it happens ---
L_SHIELD_SWEEP = "shield: swept path"
L_SHIELD_CLEAR = "shield: CLEARANCE vs robots+obstacles"
L_SHIELD_PROG  = "shield: predicted progress"
L_COLL_PRED    = "leaf: collision check (END STATE only)"
L_GAT          = "GAT actor forward (all callers)"
L_PREP         = "GAT prepare_state (all callers)"
L_ENV_STEP     = "SIM env.step (execute decision)"
L_ENV_SUBSTEP  = "SIM sim.step sub-step"
L_SIM_RAW      = "SIM MARL_SIM.step (irsim + state extract)"
L_IRSIM_DYN    = "irsim: dynamics (_objects_step)"
L_IRSIM_TREE   = "irsim: build_tree (spatial index)"
L_IRSIM_STATUS = "irsim: COLLISION + arrival check"
L_IRSIM_RENDER = "irsim: render (headless no-op?)"
L_SIM_CLEAR    = "sim: robot-obstacle clearances (reward)"
L_ENV_GAT      = "SIM env GAT forward (precise action)"
L_ENV_RESET    = "SIM env.reset (per episode)"
L_TRAIN        = "TRAIN prior distil"
L_SAVE         = "save pi shards"
L_BUILD_ENV    = "build env + backbone (once)"

ORDER = [
    L_EVAL, L_DECIDE, L_BUILD_MODEL, L_SEARCH,
    L_ROOT_COARSE, L_ROOT_PRECISE, L_SIMULATE, L_TREE_COARSE, L_TREE_PRECISE,
    L_EXPAND, L_PRIOR, L_FEATS, L_MAKE_NODE, L_COLL_PRED, L_COST_TO_GO,
    L_MODEL_RS, L_COARSE_ACT, L_SHIELD_SWEEP, L_SHIELD_CLEAR, L_SHIELD_PROG,
    L_GAT, L_PREP,
    L_ENV_STEP, L_ENV_SUBSTEP, L_SIM_RAW, L_IRSIM_DYN, L_IRSIM_TREE,
    L_IRSIM_STATUS, L_IRSIM_RENDER, L_SIM_CLEAR, L_ENV_GAT, L_ENV_RESET,
    L_TRAIN, L_SAVE, L_BUILD_ENV,
]
INDENT = {
    L_DECIDE: 1, L_BUILD_MODEL: 2, L_SEARCH: 2,
    L_ROOT_COARSE: 3, L_ROOT_PRECISE: 3, L_SIMULATE: 3,
    L_TREE_COARSE: 4, L_TREE_PRECISE: 4, L_EXPAND: 4,
    L_PRIOR: 5, L_FEATS: 6, L_MAKE_NODE: 4, L_COLL_PRED: 5, L_COST_TO_GO: 5,
    L_MODEL_RS: 4, L_COARSE_ACT: 4,
    L_SHIELD_SWEEP: 4, L_SHIELD_CLEAR: 4, L_SHIELD_PROG: 4,
    L_GAT: 4, L_PREP: 5,
    L_ENV_STEP: 1, L_ENV_SUBSTEP: 2, L_SIM_RAW: 3,
    L_IRSIM_DYN: 4, L_IRSIM_TREE: 4, L_IRSIM_STATUS: 4, L_IRSIM_RENDER: 4,
    L_SIM_CLEAR: 4, L_ENV_GAT: 2, L_ENV_RESET: 1,
}


class Gaz14Instrumentation:
    """Installs the timers and records one snapshot per plan→distil cycle."""

    def __init__(self, prof: Profiler) -> None:
        self.prof = prof
        self.phase = "root"                    # flipped by the first descent
        self.cycles: list[dict[str, Stat]] = []
        self._mark = prof.snapshot()

    # -- cycle bookkeeping -------------------------------------------------

    def close_cycle(self) -> None:
        """Record the timings since the last cycle boundary (no-op if idle)."""
        now = self.prof.snapshot()
        delta = Profiler.delta(now, self._mark)
        if any(v.calls for v in delta.values()):
            self.cycles.append(delta)
            self._mark = now

    # -- installation ------------------------------------------------------

    def install(self) -> None:
        p = self.prof

        from robot_nav import iterate_gaz14
        from robot_nav.models.MARL.capswitcher.policies.gat_backbone import (
            GATBackbone,
        )
        from robot_nav.models.MARL.capswitcher.rl.reward import PRECISE
        from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv
        from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
            CoarseSteering14,
        )
        from robot_nav.models.MARL.capswitcher_14.rl import forward_model as fm
        from robot_nav.models.MARL.capswitcher_14.rl.search import (
            features, gumbel, prior_net, tree,
        )

        # ---- the headline: one label per (phase, mode) materialisation ----
        def materialize_label(args, kwargs) -> str:
            node = kwargs.get("node", args[0] if args else None)
            a = kwargs.get("a", args[1] if len(args) > 1 else None)
            precise = (
                node is not None and a is not None
                and node.branches[a].mode == PRECISE
            )
            if self.phase == "root":
                return L_ROOT_PRECISE if precise else L_ROOT_COARSE
            return L_TREE_PRECISE if precise else L_TREE_COARSE

        # ``gumbel`` imported these by value; patch both bindings so a call
        # goes through exactly one wrapper whichever module issued it.
        for mod in (tree, gumbel):
            if hasattr(mod, "materialize"):
                p.patch(mod, "materialize", materialize_label)
            if hasattr(mod, "expand_node"):
                p.patch(mod, "expand_node", L_EXPAND)
            if hasattr(mod, "make_node"):
                p.patch(mod, "make_node", L_MAKE_NODE)

        # The root phase ends at the first descent of the decision.
        def phase_simulate(args, kwargs) -> str:
            self.phase = "tree"
            return L_SIMULATE

        for mod in (tree, gumbel):
            if hasattr(mod, "simulate"):
                p.patch(mod, "simulate", phase_simulate)

        def phase_run(args, kwargs) -> str:
            self.phase = "root"                # new decision → eager root again
            return L_SEARCH

        p.patch(gumbel.GumbelAlphaZero14, "run", phase_run)
        p.patch(gumbel.GumbelSwitcher14, "decide", L_DECIDE)
        p.patch(gumbel.GumbelSwitcher14, "_build_model", L_BUILD_MODEL)

        # ---- forward model internals --------------------------------------
        p.patch(fm.ForwardModel14, "robot_state", L_MODEL_RS)
        p.patch(fm.ForwardModel14, "cost_to_go", L_COST_TO_GO)
        p.patch(CoarseSteering14, "compute_actions", L_COARSE_ACT)

        # ---- obstacle checking #1: the shield's swept vet (coarse only) ----
        # ``forward_model`` imported these by value, so patch them there.
        p.patch(fm, "swept_positions", L_SHIELD_SWEEP)
        p.patch(fm, "min_member_clearance", L_SHIELD_CLEAR)
        p.patch(fm, "predicted_progress", L_SHIELD_PROG)

        # ---- obstacle checking #2: the planner's end-state snapshot --------
        p.patch(fm.ForwardModel14, "collision_pred", L_COLL_PRED)

        # ---- the frozen GAT (the suspected hot spot) ----------------------
        p.patch(GATBackbone, "get_embedding_and_actions", L_GAT)
        p.patch(GATBackbone, "prepare_state", L_PREP)

        # ---- prior / features ---------------------------------------------
        p.patch(prior_net.LearnedPrior, "_evaluate", L_PRIOR)
        p.patch(features.GroupFeatureBuilder, "__call__", L_FEATS)

        # ---- simulation ----------------------------------------------------
        p.patch(SwitcherEnv, "step", L_ENV_STEP)
        p.patch(SwitcherEnv, "_apply_substep", L_ENV_SUBSTEP)
        p.patch(SwitcherEnv, "_refresh_cache", L_ENV_GAT)
        p.patch(SwitcherEnv, "reset", L_ENV_RESET)

        # ---- obstacle checking #3: the real simulator ----------------------
        from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

        p.patch(MARL_SIM_OBSTACLE, "step", L_SIM_RAW)
        p.patch(MARL_SIM_OBSTACLE, "get_robot_obstacle_clearances", L_SIM_CLEAR)

        # ---- loop-level stages (bound into iterate_gaz14 at import) --------
        p.patch(iterate_gaz14, "run", L_EVAL)
        p.patch(iterate_gaz14, "_save_pi_targets", L_SAVE)

        # irsim's internals live on the env *instance*, which only exists once
        # build_env has run — so patch them on the way out of it.
        build_env = iterate_gaz14.build_env

        def timed_build_env(*args, **kwargs):
            with p.timed(L_BUILD_ENV):
                env, coarse, sim = build_env(*args, **kwargs)
            for attr, label in (
                ("_objects_step", L_IRSIM_DYN),
                ("build_tree", L_IRSIM_TREE),
                ("_objects_check_status", L_IRSIM_STATUS),
                ("render", L_IRSIM_RENDER),
            ):
                if hasattr(sim.env, attr):
                    p.patch(sim.env, attr, label)
            return env, coarse, sim

        iterate_gaz14.build_env = timed_build_env

        # Distillation closes a cycle, so snapshot right after it returns.
        train_prior = iterate_gaz14.train_prior

        def timed_train_prior(*args, **kwargs):
            try:
                with p.timed(L_TRAIN):
                    return train_prior(*args, **kwargs)
            finally:
                self.close_cycle()

        iterate_gaz14.train_prior = timed_train_prior


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _get(stats: dict[str, Stat], label: str) -> Stat:
    return stats.get(label, Stat())


def _decisions(stats: dict[str, Stat]) -> Stat:
    """``decide`` when the switcher drove the run, else the raw search calls."""
    dec = _get(stats, L_DECIDE)
    return dec if dec.calls else _get(stats, L_SEARCH)


def summary(stats: dict[str, Stat], wall: float) -> str:
    """The per-decision / per-episode read of the raw table."""
    dec = _decisions(stats)
    n_dec = dec.calls
    if not n_dec:
        return "(no decisions recorded — nothing to summarise)"
    n_ep = max(_get(stats, L_ENV_RESET).calls, 1)

    rc, rp = _get(stats, L_ROOT_COARSE), _get(stats, L_ROOT_PRECISE)
    tc, tp = _get(stats, L_TREE_COARSE), _get(stats, L_TREE_PRECISE)
    env, train = _get(stats, L_ENV_STEP), _get(stats, L_TRAIN)
    gat, prep = _get(stats, L_GAT), _get(stats, L_PREP)
    sim_total = env.total + _get(stats, L_ENV_RESET).total

    def per_dec(s: Stat) -> str:
        return (f"{1e3 * s.total / n_dec:8.1f} ms/decision "
                f"({s.calls / n_dec:5.1f} calls × {s.ms_per_call:6.2f} ms)"
                f"  [{100.0 * s.total / wall:4.1f}% wall]")

    lines = [
        "",
        f"PER-DECISION BREAKDOWN  ({n_dec} decisions over {n_ep} episodes, "
        f"{wall:.1f} s wall)",
        "-" * 96,
        f"  planning (decide)          {per_dec(dec)}",
        f"    root: coarse vets        {per_dec(rc)}",
        f"    root: precise rollout    {per_dec(rp)}",
        f"    tree: coarse vets        {per_dec(tc)}",
        f"    tree: precise rollouts   {per_dec(tp)}",
        f"  simulation (env.step)      {per_dec(env)}",
        "-" * 96,
    ]

    mat_total = rc.total + rp.total + tc.total + tp.total
    precise_total = rp.total + tp.total
    coarse_total = rc.total + tc.total
    lines += [
        "MATERIALISATION SPLIT",
        f"  precise edges   {precise_total:8.2f} s  "
        f"({100.0 * precise_total / max(mat_total, 1e-9):5.1f}% of materialisation, "
        f"{100.0 * precise_total / wall:4.1f}% of wall)  "
        f"{rp.calls + tp.calls} edges, "
        f"{1e3 * precise_total / max(rp.calls + tp.calls, 1):.2f} ms each",
        f"  coarse edges    {coarse_total:8.2f} s  "
        f"({100.0 * coarse_total / max(mat_total, 1e-9):5.1f}% of materialisation, "
        f"{100.0 * coarse_total / wall:4.1f}% of wall)  "
        f"{rc.calls + tc.calls} vets, "
        f"{1e3 * coarse_total / max(rc.calls + tc.calls, 1):.3f} ms each",
        f"  precise:coarse cost ratio per edge  "
        f"{(precise_total / max(rp.calls + tp.calls, 1)) / max(coarse_total / max(rc.calls + tc.calls, 1), 1e-12):.1f}×",
        "-" * 96,
        "OBSTACLE / COLLISION CHECKING — three different checks, three places",
    ]

    def chk(label: str, note: str) -> str:
        s = _get(stats, label)
        return (f"  {note:<44}{s.total:7.2f} s ({100.0 * s.total / wall:4.1f}%)"
                f"  {s.calls:8d} calls  {s.ms_per_call:7.3f} ms")

    lines += [
        chk(L_SHIELD_CLEAR, "coarse vet: swept clearance (exact)"),
        chk(L_SHIELD_SWEEP, "coarse vet: swept path build"),
        chk(L_COLL_PRED,    "planner leaf: end-state snapshot"),
        chk(L_IRSIM_STATUS, "irsim: collision + arrival (the real one)"),
        chk(L_IRSIM_TREE,   "irsim: spatial index for it"),
        chk(L_SIM_CLEAR,    "sim: robot-obstacle clearance (reward)"),
        f"  precise edges are NOT swept-vetted in the tree — only their end "
        f"state is checked ({_get(stats, L_ROOT_PRECISE).calls + _get(stats, L_TREE_PRECISE).calls} edges)",
        "-" * 96,
        "FROZEN GAT (the shared hot path)",
        f"  actor forwards  {gat.calls:8d} calls  {gat.total:8.2f} s  "
        f"({100.0 * gat.total / wall:4.1f}% wall)  {gat.ms_per_call:.3f} ms each"
        f"   [{_get(stats, L_ENV_GAT).total:.2f} s of it inside the simulator]",
        f"  prepare_state   {prep.calls:8d} calls  {prep.total:8.2f} s  "
        f"({100.0 * prep.total / wall:4.1f}% wall)  {prep.ms_per_call:.3f} ms each",
        "-" * 96,
        "TOP-LEVEL SPLIT",
        f"  planning        {dec.total:8.2f} s  ({100.0 * dec.total / wall:4.1f}%)"
        f"   {dec.total / n_ep:6.2f} s/episode",
        f"  simulation      {sim_total:8.2f} s  ({100.0 * sim_total / wall:4.1f}%)"
        f"   {sim_total / n_ep:6.2f} s/episode",
        f"  distillation    {train.total:8.2f} s  "
        f"({100.0 * train.total / wall:4.1f}%)   {train.calls} cycle(s), "
        f"{train.ms_per_call / 1e3:.2f} s each",
        "",
    ]
    return "\n".join(lines)


def cycle_table(cycles: list[dict[str, Stat]]) -> str:
    """Per-cycle wall split — does planning get cheaper as the prior learns?"""
    if not cycles:
        return ""
    rows = [
        ("decisions",        lambda s: f"{_decisions(s).calls:d}"),
        ("plan s",           lambda s: f"{_decisions(s).total:.1f}"),
        ("  root coarse s",  lambda s: f"{_get(s, L_ROOT_COARSE).total:.1f}"),
        ("  root precise s", lambda s: f"{_get(s, L_ROOT_PRECISE).total:.1f}"),
        ("  tree coarse s",  lambda s: f"{_get(s, L_TREE_COARSE).total:.1f}"),
        ("  tree precise s", lambda s: f"{_get(s, L_TREE_PRECISE).total:.1f}"),
        ("sim s",            lambda s: f"{_get(s, L_ENV_STEP).total:.1f}"),
        ("distil s",         lambda s: f"{_get(s, L_TRAIN).total:.1f}"),
        ("ms/decision",      lambda s: (
            f"{1e3 * _decisions(s).total / max(_decisions(s).calls, 1):.0f}"
        )),
    ]
    head = f"{'per cycle':<18}" + "".join(
        f"{f'cyc{i}':>12}" for i in range(1, len(cycles) + 1)
    )
    out = ["", "=" * len(head), head, "-" * len(head)]
    for label, fn in rows:
        out.append(f"{label:<18}" + "".join(f"{fn(c):>12}" for c in cycles))
    out.append("=" * len(head))
    return "\n".join(out)


def report(prof: Profiler, inst: Gaz14Instrumentation, wall: float) -> str:
    return "\n".join([
        Profiler.table(
            prof.stats, wall=wall, order=ORDER, indent=INDENT,
            title="CAPSwitcher-14 iterate loop — wall-clock profile",
        ),
        summary(prof.stats, wall),
        cycle_table(inst.cycles),
    ])


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Profile robot_nav.iterate_gaz14 (all other flags are "
                    "forwarded to it verbatim).",
        add_help=False,
    )
    ap.add_argument("--report-json", type=str, default=None,
                    help="also write the raw timings + per-cycle deltas here")
    ap.add_argument("--profile-help", action="help",
                    help="show this message (--help goes to iterate_gaz14)")
    args, rest = ap.parse_known_args()

    prof = Profiler()
    inst = Gaz14Instrumentation(prof)
    inst.install()

    from robot_nav import iterate_gaz14

    argv = sys.argv
    sys.argv = ["iterate_gaz14"] + rest
    t0 = time.perf_counter()
    try:
        iterate_gaz14.main()
    finally:
        wall = time.perf_counter() - t0
        sys.argv = argv
        inst.close_cycle()                      # trailing / crashed-mid cycle
        prof.unpatch()
        # Report whatever was measured, including after a crash mid-run; stay
        # silent when nothing ran (``--help``) and never swallow the exception.
        if prof.stats:
            print(report(prof, inst, wall))
            if args.report_json:
                path = Path(args.report_json)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = prof.to_dict(wall)
                payload["cycles"] = [
                    {k: {"calls": v.calls, "total_s": v.total,
                         "self_s": v.self_time}
                     for k, v in c.items() if v.calls}
                    for c in inst.cycles
                ]
                path.write_text(json.dumps(payload, indent=2))
                print(f"Wrote {path}")


if __name__ == "__main__":
    main()
