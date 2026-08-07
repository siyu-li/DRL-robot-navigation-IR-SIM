"""
Watch a trained CAPSwitcher-14 prior plan, one episode at a time.

Same env, same seeds and the same ``GumbelSwitcher14`` as ``eval_mpc_14`` /
``iterate_gaz14`` — only plotting is on and the per-decision trace is printed,
so episode *k* here is exactly episode *k* of the evaluation table.

Each decision prints its mode, the chosen coarse group (with members), what the
decision cost, and the running episode cost split into its coarse and precise
components.  Terminal events name the flagged robots and, for precise, the
robot that was being driven at the time — the collision-attribution question.

Usage (GPU box; irsim's step is what needs the machine, plotting is extra):
    # watch live in a window (needs a display / X forwarding)
    python -m robot_nav.render_gaz14 \
        --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
        --value-model <value ckpt> --episodes 3

    # headless box: write a GIF per episode instead
    python -m robot_nav.render_gaz14 \
        --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
        --value-model <value ckpt> --episodes 3 --save-ani --out-dir renders

    # jump straight to an episode the eval table flagged (same --seed!)
    python -m robot_nav.render_gaz14 --prior-model <ckpt> --only-episode 17
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.eval_mpc_14 import (
    COARSE,
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_COST_CONFIG,
    build_env,
    resolve_device,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
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
    random.seed(seed)
    np.random.seed(seed)
    env._coarse_rng = np.random.default_rng(seed)

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
    ap.add_argument("--prior-model", type=str, required=True,
                    help="PriorNet checkpoint, e.g. "
                         "runs/gaz14_value/cycle_05_prior/prior_best.pt")
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
    ap.add_argument("--save-ani", action="store_true",
                    help="write an animation instead of showing a window "
                         "(headless-friendly)")
    ap.add_argument("--out-dir", type=str, default="renders")
    ap.add_argument("--quiet", action="store_true",
                    help="omit the per-decision trace")
    # --- search hyper-parameters: keep identical to the eval run ---
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--feas-margin", type=float, default=0.0)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt,
        disable_plotting=False,          # the whole point of this script
        save_ani=args.save_ani,
    )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)

    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)
    from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
        LearnedPrior,
        PriorNet,
    )

    prior = LearnedPrior(
        PriorNet.load(args.prior_model, map_location=device),
        feature_builder,
        feas_margin=args.feas_margin,
        device=device,
    )
    policy = GumbelSwitcher14(
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
        f"precise_unit={cost.precise_unit}, budget={args.budget}, "
        f"device={device}\nPrior: {args.prior_model}\n"
        f"Leaf: {args.value_model or 'analytic'}"
    )

    eps = (
        [args.only_episode] if args.only_episode is not None
        else list(range(args.episodes))
    )
    out_dir = Path(args.out_dir)
    if args.save_ani:
        # save_animate() writes to the global path manager's ani_path; it takes
        # no path argument, so point it at --out-dir here.
        from irsim.config.path_param import path_manager

        out_dir.mkdir(parents=True, exist_ok=True)
        path_manager.ani_path = str(out_dir)

    summaries = []
    for ep in eps:
        summaries.append(render_episode(
            env, policy, coarse, args.seed + ep, ep,
            args.render_delay, not args.quiet,
        ))
        if args.save_ani:
            # IRSim buffers frames for the whole session, so flush per episode
            # to get one file each.  Not env.end() — that tears the env down
            # (closes figures, resets the object registry) and the next episode
            # would have nothing to run in.  rm_fig_path clears the buffer so
            # episodes don't bleed into each other.
            sim.env._env_plot.save_animate(
                ani_name=f"gaz14_ep{ep:03d}", rm_fig_path=True,
            )

    if not args.save_ani:
        sim.env.end(ending_time=0)

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    print(f"{'ep':>4} {'seed':>6} {'outcome':>10} {'decisions':>10} "
          f"{'cost':>10} {'coarse':>10} {'precise':>10}")
    for s in summaries:
        print(f"{s['episode']:>4} {s['seed']:>6} {s['outcome']:>10} "
              f"{s['decisions']:>10} {s['cost']:>10.0f} "
              f"{s['coarse_cost']:>10.0f} {s['precise_cost']:>10.0f}")
    if args.save_ani:
        print(f"\nAnimations written to {out_dir}/")
    print()


if __name__ == "__main__":
    main()
