"""
Classical plan-to-goal baselines for the 14-robot switcher: A*, LevinTS,
PHS_h, PHS* — the comparison set for GAZ14-L.

Each baseline is the canonical best-first algorithm (Orseau & Lelis, AAAI
2021 — see ``capswitcher_14/rl/search/best_first.py`` for the exact φ's and
the domain mapping): it plans from the episode's start state to an in-model
``all_reached`` node over the same :class:`ForwardModel14`, same
:class:`SwitcherCost` step costs, same 22-groups+precise action set, then
replays the plan through the env (re-planning only when the plan runs out
before the episode ends — model–sim drift).  Guidance comes from the same
learned functions GAZ uses:

* h  = ``--value-model`` (:class:`LearnedCostToGo`, per-robot value summed
  over unreached robots, priced by the live cost table); analytic
  α·Σdist fallback when absent.  Used by A*, PHS_h, PHS*.
* π  = ``--prior-model`` (:class:`LearnedPrior` logits over the 23 stubs);
  uniform fallback.  Used by LevinTS, PHS_h, PHS*.

Search effort is reported in **materialised edges** — the model's own
``n_coarse_vets`` (refuted vets included) and ``n_precise_expansions``
counters, per episode — rather than a per-decision budget: plan-to-goal
concentrates its transitions in one call, so GAZ's per-decision budget knob
does not apply.  ``--max-transitions`` caps a single planning call; a cap-hit
executes the best generated goal plan (still exact in-model) or the best
partial path by g + h, and is counted in the table.

Episodes, seeds, metrics and the result table are shared with
``eval_gaz14_lazy`` (same ``build_env`` / ``run`` / ``print_table``), so rows
are comparable across the two scripts' tables at matched ``--seed``.

Usage (run on the GPU box — local irsim step crashes; see project memory):
    python -m robot_nav.eval_gaz14_baselines --episodes 100 \
        --algos astar levints phs phs-star \
        --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
        --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt
"""

from __future__ import annotations

import argparse

import numpy as np
from loguru import logger

from robot_nav.eval_gaz14_lazy import (
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_COST_CONFIG,
    RESULT_ROWS,
    add_layout_args,
    build_env,
    layout_from_args,
    print_table,
    resolve_device,
    run,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.search.best_first import (
    EVALUATIONS,
    PlanToGoalSwitcher14,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)

logger.disable("irsim")

# Pretty table names for the CLI algo keys.
ALGO_LABELS = {
    "astar": "A*",
    "levints": "LevinTS",
    "levints-depth": "LevinTS-d",
    "phs": "PHS",
    "phs-star": "PHS*",
}

# Search-effort rows appended below the shared outcome metrics.  Effort is in
# materialised edges (model transition counters), averaged per episode.
BASELINE_ROWS = RESULT_ROWS + [
    ("plans/ep",             "avg_plans",            "{:.2f}"),
    ("plan solved rate",     "plan_solved_rate",     "{:.1%}"),
    ("cap hits",             "cap_hits",             "{:d}"),
    ("fallback decisions",   "fallbacks",            "{:d}"),
    ("coarse vets/ep",       "avg_coarse_vets",      "{:.0f}"),
    ("precise rollouts/ep",  "avg_precise_rollouts", "{:.0f}"),
    ("transitions/ep",       "avg_transitions_ep",   "{:.0f}"),
    ("expansions/ep",        "avg_expansions",       "{:.0f}"),
]


def _episode_tracker(policy: PlanToGoalSwitcher14):
    """
    ``on_step`` observer diffing the policy's cumulative counters at each
    episode end — per-episode search effort from the same loop that produces
    the outcome metrics.  Prints one progress line per finished episode:
    plan-to-goal runs are long, and a silent 12-hour loop is undebuggable.
    """
    per_ep: list[dict] = []
    prev = policy.snapshot()

    def on_step(ep, decision, step_cost, info, done) -> None:
        nonlocal prev
        if done:
            policy.reset_plan()      # never replay a stale suffix next episode
            cur = policy.snapshot()
            d = {k: cur[k] - prev[k] for k in cur}
            per_ep.append(d)
            prev = cur
            outcome = (
                "SUCCESS" if info.get("all_reached")
                else "COLLISION" if info.get("collision")
                else "TIMEOUT" if info.get("timeout") else "ENDED"
            )
            print(
                f"  ep {ep:3d}: {outcome:<9} plans={d['plans']:<3} "
                f"solved={d['solved_plans']:<3} cap_hits={d['cap_hits']:<3} "
                f"transitions={d['coarse_vets'] + d['precise_expansions']}",
                flush=True,
            )

    return on_step, per_ep


def _effort_stats(per_ep: list[dict]) -> dict:
    plans = sum(e["plans"] for e in per_ep)
    return {
        "avg_plans": float(np.mean([e["plans"] for e in per_ep])),
        "plan_solved_rate": (
            sum(e["solved_plans"] for e in per_ep) / plans if plans else 0.0
        ),
        "cap_hits": sum(e["cap_hits"] for e in per_ep),
        "fallbacks": sum(e["fallbacks"] for e in per_ep),
        "avg_coarse_vets": float(np.mean([e["coarse_vets"] for e in per_ep])),
        "avg_precise_rollouts": float(
            np.mean([e["precise_expansions"] for e in per_ep])
        ),
        "avg_transitions_ep": float(
            np.mean([e["coarse_vets"] + e["precise_expansions"] for e in per_ep])
        ),
        "avg_expansions": float(np.mean([e["expansions"] for e in per_ep])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--algos", type=str, nargs="+",
                    default=["astar", "levints", "phs", "phs-star"],
                    choices=sorted(EVALUATIONS),
                    help="best-first baselines to run (see best_first.py)")
    ap.add_argument("--max-transitions", type=int, default=20000,
                    help="cap on model transitions (coarse vets + precise "
                         "rollouts) per planning call; a cap-hit falls back "
                         "to the best plan found so far")
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--prior-model", type=str, default=None,
                    help="learned PriorNet checkpoint -> π for LevinTS/PHS/"
                         "PHS*; default uniform")
    ap.add_argument("--value-model", type=str, default=None,
                    help="learned cost-to-go checkpoint -> h for A*/PHS/PHS*; "
                         "default analytic α·Σdist")
    add_layout_args(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    layout = layout_from_args(args)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt, layout=layout,
    )
    print(
        f"World: {'corridor ' + str(layout.band) if layout else 'scattered'}\n"
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"{len(MOVE_GROUPS)} coarse groups, cost_config={args.cost_config}, "
        f"d_safe={args.d_safe}, max_transitions={args.max_transitions}, "
        f"prior={args.prior_model or 'uniform'}, "
        f"value={args.value_model or 'analytic'}, device={device}"
    )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        print(f"Learned leaf: feature={leaf_value.feature}")

    prior = None
    if args.prior_model:
        from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
            LearnedPrior,
            PriorNet,
        )

        prior = LearnedPrior(
            PriorNet.load(args.prior_model, map_location=device),
            GroupFeatureBuilder(MOVE_GROUPS),
            device=device,
        )
        print(f"Learned prior: {args.prior_model}")

    results: dict[str, dict] = {}
    for algo in args.algos:
        policy = PlanToGoalSwitcher14(
            backbone=env.backbone,
            coarse=coarse,
            sim=sim,
            evaluate=EVALUATIONS[algo],
            prior=prior,
            max_transitions=args.max_transitions,
            d_safe=args.d_safe,
            selection_interval=env.selection_interval,
            goal_threshold=args.goal_threshold,
            cost=env.cost,
            leaf_value=leaf_value,
        )
        name = ALGO_LABELS[algo]
        print(f"\nRunning {name} for {args.episodes} episodes ...")
        on_step, per_ep = _episode_tracker(policy)
        decide = lambda env_, p=policy: p.decide(env_._robot_state)  # noqa: E731
        stats = run(env, decide, args.episodes, args.seed,
                    policy=policy, on_step=on_step)
        stats.update(_effort_stats(per_ep))
        results[name] = stats
        # Print each algorithm's rows the moment it finishes — these runs take
        # hours per algorithm, and a killed run must not lose finished results.
        print_table({name: stats}, rows=BASELINE_ROWS)

    if len(results) > 1:
        print_table(results, rows=BASELINE_ROWS)


if __name__ == "__main__":
    main()
