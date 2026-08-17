"""
Watch a CAPSwitcher-14 policy plan, one episode at a time — either variant.

Same env, same seeds and the same switcher as the evaluation harnesses — only
plotting is on and the per-decision trace is printed, so episode *k* here is
exactly episode *k* of the evaluation table:

* **lazy** (default) — ``GumbelSwitcher14`` + a learned ``PriorNet``, as in
  ``eval_gaz14_lazy`` / ``iterate_gaz14``.  ``--prior-model`` is required.
* **eager** (``--eager``) — ``GumbelSwitcher14Eager`` + ``HeuristicPrior14``,
  as in ``eval_gaz14_eager``.  No checkpoint needed (the eager prior reads the
  vet instead of being learned); ``--budget`` / ``--m`` default to that
  harness's 115 / 4 rather than the lazy 100 / 16.

That equality relies on ``seed_episode`` being called before every ``reset()``:
on the pinned ir-sim 2.x the obstacle layout and the goals are drawn from the
``np.random`` legacy global, so seeding it is what pins the world, not just the
robot start poses.  Runs made before that fix cannot be replayed — their
layouts were drawn from entropy and are gone.

Each decision prints its mode, the chosen coarse group (with members), what the
decision cost, and the running episode cost split into its coarse and precise
components.  Terminal events name the flagged robots and, for precise, the
robot that was being driven at the time — the collision-attribution question.

Live viewing only — nothing is written to disk.  A display is required (X
forwarding when running on the GPU box).

Usage (GPU box; irsim's step is what needs the machine, plotting is extra):
    # lazy, learned prior
    python -m robot_nav.render_gaz14 \
        --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
        --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt --episodes 3

    # eager, heuristic prior — the GAZ14-E row
    python -m robot_nav.render_gaz14 --eager --budget 115 \
        --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt --episodes 3

    # jump straight to an episode the eval table flagged (same --seed!)
    python -m robot_nav.render_gaz14 --prior-model <ckpt> --only-episode 17
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from loguru import logger

from robot_nav.eval_gaz14_lazy import (
    COARSE,
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_COST_CONFIG,
    add_layout_args,
    build_env,
    layout_from_args,
    resolve_device,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.switcher_env import seed_episode
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import GumbelSwitcher14

logger.disable("irsim")


def _outcome(info: dict) -> str:
    """One-word episode outcome from the terminal info dict."""
    if info.get("all_reached"):
        return "SUCCESS"
    if info.get("collision"):
        return "COLLISION"
    if info.get("timeout"):
        return "TIMEOUT"
    return "ENDED"


def render_episode(
    env, policy: GumbelSwitcher14, coarse, seed: int, ep: int,
    render_delay: float, verbose: bool,
) -> dict:
    """Run one seeded episode with plotting on; return its summary."""
    seed_episode(env, seed)

    env.reset()
    done = False
    step = 0
    ep_cost = ep_coarse = ep_precise = 0.0
    info: dict = {}

    print(f"\n{'=' * 78}\nEpisode {ep} (seed {seed})\n{'=' * 78}")
    while not done:
        decision = policy.decide(env._robot_state)
        _, _, done, info = env.step(
            decision["mode"], group=decision["group"], frames=decision["frames"]
        )
        step += 1
        cost = float(info["path_cost"])
        ep_cost += cost
        if decision["mode"] == COARSE:
            ep_coarse += cost
        else:
            ep_precise += cost

        if verbose:
            if decision["mode"] == COARSE:
                g = info["group"]
                members = (
                    list(coarse.members_of(g)) if g is not None else []
                )
                what = f"COARSE  g{g:<2} members={members}"
            else:
                what = f"PRECISE robots_moved={info['robots_moved']:<2}"
            print(
                f"  [{step:3d}] {what:<52} cost {cost:8.1f}  "
                f"cum {ep_cost:9.1f}"
            )
        if render_delay:
            time.sleep(render_delay)

    outcome = _outcome(info)
    detail = ""
    if info.get("collision"):
        flagged = info.get("collision_robots") or []
        if decision["mode"] == COARSE:
            detail = (
                f" — during COARSE group {info['group']}, flagged {flagged} "
                "(shield breach: this should not happen)"
            )
        else:
            active = info.get("active_robot")
            who = (
                f"the DRIVEN robot {active}"
                if active in set(flagged)
                else f"a STATIONARY bystander (driven was {active})"
            )
            detail = f" — during PRECISE, flagged {flagged}: {who}"

    print(
        f"  -> {outcome}{detail}\n"
        f"     {step} decisions   cost {ep_cost:.0f} "
        f"(coarse {ep_coarse:.0f} + precise {ep_precise:.0f}, "
        f"precise share {ep_precise / max(ep_cost, 1e-9):.1%})"
    )
    return {
        "episode": ep, "seed": seed, "outcome": outcome, "decisions": step,
        "cost": ep_cost, "coarse_cost": ep_coarse, "precise_cost": ep_precise,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", action="store_true",
                    help="render GAZ14-E (GumbelSwitcher14Eager + "
                         "HeuristicPrior14) instead of the lazy switcher")
    ap.add_argument("--prior-model", type=str, default=None,
                    help="PriorNet checkpoint, e.g. "
                         "runs/gaz14_value/cycle_05_prior/prior_best.pt "
                         "(lazy only, and required there)")
    ap.add_argument("--value-model", type=str, default=None,
                    help="leaf cost-to-go checkpoint — pass the SAME one the "
                         "run was planned with, or the search differs")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1000,
                    help="must match the eval run to replay its episodes")
    ap.add_argument("--only-episode", type=int, default=None,
                    help="render just this episode index (0-based, as in eval)")
    ap.add_argument("--render-delay", type=float, default=0.0,
                    help="extra seconds per decision, to watch it more slowly")
    ap.add_argument("--quiet", action="store_true",
                    help="omit the per-decision trace")
    # --- search hyper-parameters: keep identical to the eval run ---
    # Defaults are per-variant (lazy 100/16, eager 115/4, as in the two eval
    # harnesses), so they are resolved after parsing rather than declared here.
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--feas-margin", type=float, default=0.0)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto")
    # --- eager-only prior knobs (see priors_eager.HeuristicPrior14) ---
    ap.add_argument("--uniform-prior", action="store_true",
                    help="eager only: flat logits instead of HeuristicPrior14")
    ap.add_argument("--w-eff", type=float, default=1.0)
    ap.add_argument("--w-clear", type=float, default=0.5)
    ap.add_argument("--precise-bias", type=float, default=0.0)
    ap.add_argument("--prior-temperature", type=float, default=1.0)
    ap.add_argument("--z-clip", type=float, default=3.0)
    add_layout_args(ap)
    args = ap.parse_args()

    if not args.eager and not args.prior_model:
        raise SystemExit(
            "--prior-model is required for the lazy switcher (its prior is "
            "learned).  Pass --eager to render GAZ14-E, whose HeuristicPrior14 "
            "needs no checkpoint."
        )
    if args.budget is None:
        args.budget = 115 if args.eager else 100
    if args.m is None:
        args.m = 4 if args.eager else 16

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    layout = layout_from_args(args)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt,
        disable_plotting=False,          # the whole point of this script
        layout=layout,
    )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)

    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)
    if args.eager:
        from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel_eager import (
            GumbelSwitcher14Eager,
        )
        from robot_nav.models.MARL.capswitcher_14.rl.search.priors_eager import (
            HeuristicPrior14,
            UniformPrior,
        )

        from robot_nav.eval_gaz14_eager import check_budget

        # Same guard the eval harness applies: an unspendable remainder here
        # would mean rendering a shallower search than the row being replayed.
        check_budget(args.budget, len(coarse.selectable_groups()) + 1, args.m)
        switcher_cls = GumbelSwitcher14Eager
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
        prior_desc = "uniform" if args.uniform_prior else "HeuristicPrior14"
    else:
        from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
            LearnedPrior,
            PriorNet,
        )

        switcher_cls = GumbelSwitcher14
        prior = LearnedPrior(
            PriorNet.load(args.prior_model, map_location=device),
            feature_builder,
            feas_margin=args.feas_margin,
            device=device,
        )
        prior_desc = args.prior_model

    policy = switcher_cls(
        backbone=env.backbone,
        coarse=coarse,
        sim=sim,
        prior=prior,
        budget=args.budget,
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
    print(
        f"Env: {sim.num_robots} robots, {len(MOVE_GROUPS)} coarse groups, "
        f"precise_unit={cost.precise_unit}, budget={args.budget}, m={args.m}, "
        f"device={device}\n"
        f"Variant: {'GAZ14-E (eager)' if args.eager else 'GAZ14-L (lazy)'}\n"
        f"World: {'corridor ' + str(layout.band) if layout else 'scattered'}\n"
        f"Prior: {prior_desc}\n"
        f"Leaf: {args.value_model or 'analytic'}"
    )

    eps = (
        [args.only_episode] if args.only_episode is not None
        else list(range(args.episodes))
    )
    summaries = []
    for ep in eps:
        summaries.append(render_episode(
            env, policy, coarse, args.seed + ep, ep,
            args.render_delay, not args.quiet,
        ))

    sim.env.end(ending_time=0)

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    print(f"{'ep':>4} {'seed':>6} {'outcome':>10} {'decisions':>10} "
          f"{'cost':>10} {'coarse':>10} {'precise':>10}")
    for s in summaries:
        print(f"{s['episode']:>4} {s['seed']:>6} {s['outcome']:>10} "
              f"{s['decisions']:>10} {s['cost']:>10.0f} "
              f"{s['coarse_cost']:>10.0f} {s['precise_cost']:>10.0f}")
    print()


if __name__ == "__main__":
    main()
