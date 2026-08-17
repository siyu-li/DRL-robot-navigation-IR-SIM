"""
Evaluation + collection harness for the **eager** 14-robot Gumbel switcher
(GAZ14-E) — the budget-matched counterpart of ``eval_gaz14_lazy.py`` (GAZ14-L).

Runs ``GumbelSwitcher14Eager``: every expanded node vets all 22 coarse groups
and rolls out precise, so every legal edge carries a real Q and nothing is
priced by a completion estimate.  Affordable because the nonlinear rotation
solve got cheap (``CoarseSteering14`` ``bfgs`` / ``bfgs_lean``); the price is
breadth — one expansion costs 23 transitions, so a budget buys ``budget // 23``
nodes against GAZ14-L's ``budget`` single edges.

Env construction, the episode loop, the metric definitions and the shard writer
are **imported from** ``eval_gaz14_lazy`` rather than reimplemented, so the two
variants are measured by identical code over identical seeded episodes.  Result
rows are tagged ``GAZ14e-b<budget>`` against ``eval_gaz14_lazy``'s ``GAZ14-b<budget>``.

Budgets should be multiples of 23 or the remainder is unspendable (expansions
are atomic).  Reference points:

    23  → root only = exhaustive depth-1 minimin ("vet everything, act greedily")
    115 → root + 4 interior nodes
    253 → root + 10

Usage (run on the GPU box — local irsim step crashes; see project memory):
    # heuristic prior, the budget sweep the two variants are compared over
    python -m robot_nav.eval_gaz14_eager --episodes 100 --budgets 23 46 115 253

    # matched GAZ14-L run for the same grid
    python -m robot_nav.eval_gaz14_lazy --episodes 100 --budgets 23 46 115 253

    # ablate the prior away, or harvest distillation targets
    python -m robot_nav.eval_gaz14_eager --episodes 100 --budgets 115 --uniform-prior
    python -m robot_nav.eval_gaz14_eager --episodes 100 --budgets 115 \
        --log-pi-targets data/pi_targets_14e
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from robot_nav.eval_gaz14_lazy import (
    DEFAULT_BACKBONE_CKPT,
    add_layout_args,
    layout_from_args,
    DEFAULT_COST_CONFIG,
    _coarse_only,
    _gumbel_decider,
    _precise_only,
    _save_pi_targets,
    build_env,
    print_table,
    resolve_device,
    run,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel_eager import (
    GumbelSwitcher14Eager,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.priors_eager import (
    HeuristicPrior14,
    UniformPrior,
)

logger.disable("irsim")


def check_budget(budget: int, expansion_cost: int, m: int) -> None:
    """
    Warn about budgets the eager search cannot actually spend.

    Two distinct failure modes, both silent otherwise: a budget that does not
    divide by the expansion cost leaves an unspendable remainder, and a budget
    that funds fewer interior expansions than ``m`` root arms leaves Sequential
    Halving with nothing to allocate (it degenerates to one visit per arm).
    """
    if budget < expansion_cost:
        raise SystemExit(
            f"--budgets {budget} is below one expansion ({expansion_cost} "
            f"transitions).  The root's vet always runs, so the run would "
            f"report a budget it did not honour.  Use {expansion_cost} or more."
        )
    if budget % expansion_cost:
        print(
            f"  NOTE: budget {budget} is not a multiple of the expansion cost "
            f"{expansion_cost}; {budget % expansion_cost} transition(s) are "
            f"unspendable (expansions are atomic).  Effective budget: "
            f"{(budget // expansion_cost) * expansion_cost}."
        )
    interior = budget // expansion_cost - 1
    if interior < m:
        print(
            f"  NOTE: budget {budget} funds {interior} interior expansion(s) "
            f"for m={m} root arms — Sequential Halving has nothing to "
            f"allocate.  Use m <= {max(interior, 1)} or raise the budget to "
            f"{(m + 1) * expansion_cost}."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[115],
                    help="transition budgets per decision; multiples of the "
                         "expansion cost (22 vets + 1 rollout = 23)")
    ap.add_argument("--m", type=int, default=4,
                    help="Gumbel-top-m root actions sampled without "
                         "replacement (lower than GAZ14-L's 16: an eager "
                         "budget funds far fewer visits to spread over arms)")
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--baselines", action="store_true",
                    help="also run precise-only and coarse-only baselines")
    ap.add_argument("--value-model", type=str, default=None,
                    help="learned cost-to-go checkpoint (train_value.py)")
    ap.add_argument("--log-pi-targets", type=str, default=None,
                    help="directory for the harvested prior-training shards "
                         "(same layout as GAZ14-L's, so train_prior.py can "
                         "consume either variant's teacher)")
    # --- heuristic prior knobs (see priors_eager.HeuristicPrior14) ---
    ap.add_argument("--uniform-prior", action="store_true",
                    help="ablate the heuristic prior away (flat logits)")
    ap.add_argument("--w-eff", type=float, default=1.0,
                    help="weight of the standardised progress-per-cost score")
    ap.add_argument("--w-clear", type=float, default=0.5,
                    help="weight of the centred clearance margin")
    ap.add_argument("--precise-bias", type=float, default=0.0)
    ap.add_argument("--prior-temperature", type=float, default=1.0)
    ap.add_argument("--z-clip", type=float, default=3.0,
                    help="soft saturation level of the efficiency score; raise "
                         "to resolve the cheap size-7 cluster's tail")
    add_layout_args(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    layout = layout_from_args(args)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt, layout=layout,
    )

    expansion_cost = len(coarse.selectable_groups()) + 1
    prior = (
        UniformPrior() if args.uniform_prior
        else HeuristicPrior14(
            w_eff=args.w_eff,
            w_clear=args.w_clear,
            precise_bias=args.precise_bias,
            temperature=args.prior_temperature,
            z_clip=args.z_clip,
            d_safe=args.d_safe,
        )
    )

    print(
        f"World: {'corridor ' + str(layout.band) if layout else 'scattered'}\n"
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"{len(MOVE_GROUPS)} coarse groups, cost_config={args.cost_config}, "
        f"precise_unit={cost.precise_unit}, d_safe={args.d_safe}, "
        f"budgets={args.budgets}, device={device}\n"
        f"GAZ14-E (eager): expansion cost = {expansion_cost} transitions "
        f"(22 vets + 1 precise rollout), solver="
        f"{coarse.nonlinear_solver}, prior="
        f"{'uniform' if args.uniform_prior else 'HeuristicPrior14'}"
    )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        print(f"Learned leaf: feature={leaf_value.feature}, "
              f"precise_cost={leaf_value.precise_cost}")

    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)
    results: dict[str, dict] = {}

    if args.baselines:
        print(f"\nRunning precise-only for {args.episodes} episodes ...")
        results["precise"] = run(env, _precise_only, args.episodes, args.seed)
        print(f"Running coarse-only for {args.episodes} episodes ...")
        results["coarse"] = run(env, _coarse_only, args.episodes, args.seed)

    for b in args.budgets:
        check_budget(b, expansion_cost, args.m)
        policy = GumbelSwitcher14Eager(
            backbone=env.backbone,
            coarse=coarse,
            sim=sim,
            prior=prior,
            budget=b,
            m=args.m,
            c_visit=args.c_visit,
            c_scale=args.c_scale,
            gumbel_scale=args.gumbel_scale,
            seed=args.seed,
            d_safe=args.d_safe,
            selection_interval=env.selection_interval,
            goal_threshold=args.goal_threshold,
            cost=env.cost,
            leaf_value=leaf_value,
            feature_builder=feature_builder,
        )
        name = f"GAZ14e-b{b}" + ("" if args.uniform_prior else "+h")
        pi_log = [] if args.log_pi_targets else None
        nodes = b // expansion_cost
        print(f"\nRunning {name} for {args.episodes} episodes "
              f"({nodes} node expansions/decision) ...")
        results[name] = run(env, _gumbel_decider(policy, pi_log),
                            args.episodes, args.seed, policy=policy)
        if pi_log:
            _save_pi_targets(pi_log, Path(args.log_pi_targets), name, args.d_safe)

    print_table(results)
    # Transitions are the budget unit, but a vet and a GAT rollout are wildly
    # different wall-clock; profile_gaz14.py measures the per-call means needed
    # to also compare the variants time-matched.
    for name, r in results.items():
        if r["avg_transitions"]:
            print(f"{name}: {r['avg_transitions']:.1f} transitions/decision "
                  f"= {r['avg_transitions'] / expansion_cost:.1f} expansions")
    if any(r["coarse_breach"] > 0 for r in results.values()):
        print("WARNING: coarse breaches > 0 — raise --d-safe or check geometry.\n")


if __name__ == "__main__":
    main()
