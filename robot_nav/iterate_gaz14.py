"""
Plan → distil loop for the 14-robot Gumbel AlphaZero switcher.

Each *cycle* runs the search over a fixed set of seeded episodes (logging the
root π′ / clearance labels), then distils every shard collected so far into a
new prior.  Cycle 1 plans with the uniform prior; cycle *t* plans with the
prior trained at the end of cycle *t−1*.  A comparison table across all cycles
is printed at the end.

The **leaf heuristic is static** — the search's cost-to-go never learns here.
Two choices:

* default — the analytic heuristic ``α · Σ_{i unreached} ‖p_i − goal_i‖`` with
  ``α = precise_unit / (lin_max · step_time)``;
* ``--value-model`` — the ``train_value.py`` per-robot regressor, summed over
  unreached robots (:class:`LearnedCostToGo`).  Its per-robot decision price is
  read live from the cost table, so an old checkpoint cannot impose stale
  pricing.

Every cycle replays the *same* episodes (``--seed`` + episode index), so rows
are directly comparable; only the prior changes between them.

Usage (GPU box):
    # analytic leaf
    python -m robot_nav.iterate_gaz14 --iterations 5 --episodes 100 \
        --budget 100 --out runs/gaz14_analytic

    # learned leaf (single-robot value net, summed over robots)
    python -m robot_nav.iterate_gaz14 --iterations 5 --episodes 100 \
        --budget 100 --value-model <ckpt> --out runs/gaz14_value
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

from robot_nav.eval_mpc_14 import (
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_COST_CONFIG,
    RESULT_ROWS,
    _gumbel_decider,
    _save_pi_targets,
    build_env,
    resolve_device,
    run,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import GumbelSwitcher14
from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
    LearnedPrior,
    PriorNet,
)
from robot_nav.train_prior import train_prior

logger.disable("irsim")

# Reported per cycle: the shared eval metrics plus this loop's distil stats.
_ROWS = RESULT_ROWS + [
    ("prior val KL",        "val_kl",          "{:.4f}"),
    ("prior feas acc",      "feas_acc",        "{:.3f}"),
    ("distil samples",      "n_samples",       "{:d}"),
    ("cycle minutes",       "minutes",         "{:.1f}"),
]


def print_table(cycles: list[dict], leaf: str) -> None:
    """Print the across-cycle comparison table."""
    names = [c["name"] for c in cycles]
    header = f"{'metric':<24}" + "".join(f"{nm:>14}" for nm in names)
    print("\n" + "=" * len(header))
    print(f"CAPSwitcher-14 plan→distil loop — leaf heuristic: {leaf}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, key, fmt in _ROWS:
        line = f"{label:<24}"
        for c in cycles:
            v = c.get(key)
            line += f"{'—' if v is None else fmt.format(v):>14}"
        print(line)
    print("-" * len(header))

    first, last = cycles[0], cycles[-1]
    d_succ = last["success_rate"] - first["success_rate"]
    print(f"cycle 1 → {len(cycles)}:  success {d_succ:+.1%}")

    # Headline cost delta on solved episodes only.  A cold cycle 1 can solve
    # nothing, so anchor on the earliest cycle that actually has a value.
    solved = [(i, c) for i, c in enumerate(cycles, 1) if c["avg_cost_success"]]
    if len(solved) >= 2:
        (i0, a), (i1, b) = solved[0], solved[-1]
        c0, c1 = a["avg_cost_success"], b["avg_cost_success"]
        d = c1 - c0
        note = "" if a is first else (
            f"  (earlier cycles solved nothing; anchored on {i0} → {i1})"
        )
        print(f"{'':13}cost/success ep {d:+.0f} ({d / c0:+.1%})   "
              f"[{c0:.0f} → {c1:.0f}]{note}")
    else:
        print(f"{'':13}cost/success ep — (fewer than two cycles solved anything)")

    # Precision is the term that differs between cost tables; when it dominates,
    # the total-cost trend tracks pricing rather than plan quality.
    s0, s1 = first["precise_cost_share"], last["precise_cost_share"]
    print(f"{'':13}precise cost share {s0:.1%} → {s1:.1%}")
    if s1 > 0.5:
        print("NOTE: the precise surcharge is over half of episode cost — total "
              "cost now tracks precision pricing more than plan quality. "
              "Compare cost/success ep, and consider lowering precise_unit.")

    if any(c["coarse_breach"] > 0 for c in cycles):
        print("WARNING: coarse breaches > 0 — raise --d-safe or check geometry.")
    if any(c["precise_breach_bystander"] > 0 for c in cycles):
        print("WARNING: a stationary robot was flagged in a precise collision — "
              "expected only for the driven robot; check collision semantics.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5,
                    help="plan→distil cycles (each = one eval row + one distil)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=100,
                    help="transition budget per decision (root's eager 23 count)")
    ap.add_argument("--out", type=str, required=True,
                    help="run directory: per-cycle shards + prior checkpoints")
    # --- search hyper-parameters (fixed across cycles) ---
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
    # --- static leaf heuristic ---
    ap.add_argument("--value-model", type=str, default=None,
                    help="train_value.py checkpoint used as a STATIC leaf "
                         "heuristic (per-robot values summed over unreached "
                         "robots); omit for the analytic heuristic")
    # --- distillation ---
    ap.add_argument("--prior-epochs", type=int, default=200)
    ap.add_argument("--prior-lr", type=float, default=1e-3)
    ap.add_argument("--prior-hidden", type=int, default=128)
    ap.add_argument("--w-feas", type=float, default=1.0)
    ap.add_argument("--no-replay-mix", action="store_true",
                    help="distil only the newest cycle's shards (default mixes "
                         "every cycle collected so far — the replay guard)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt,
    )

    # ---- Static leaf heuristic (never retrained inside the loop) ----------
    leaf_value = None
    leaf_desc = "analytic  α·Σ dist (unreached)"
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import (
            LearnedCostToGo,
            load_value_checkpoint,
        )

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        _, ckpt = load_value_checkpoint(args.value_model, device=device)
        meta = ckpt.get("meta", {}) or {}
        trained_n = meta.get("num_robots")
        leaf_desc = f"learned  Σ_i v_ψ(i)  [{leaf_value.feature}]"
        print(
            f"Static learned leaf: {args.value_model}\n"
            f"  feature={leaf_value.feature}  trained_num_robots={trained_n}  "
            f"per-robot decision price="
            f"{cost.precise_unit * env.selection_interval:.2f} (live cost table)"
        )
        if trained_n is not None and int(trained_n) != sim.num_robots:
            print(
                f"  NOTE: checkpoint was trained on {trained_n}-robot episodes "
                f"but this world has {sim.num_robots} — the per-robot regressor "
                "applies dimensionally, but its congestion model is "
                "out-of-distribution here."
            )

    print(
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"{len(MOVE_GROUPS)} coarse groups, precise_unit={cost.precise_unit}, "
        f"budget={args.budget}, episodes={args.episodes}, "
        f"iterations={args.iterations}, device={device}\n"
        f"Leaf heuristic (static): {leaf_desc}"
    )

    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)
    prior = None                     # cycle 1 plans with the uniform prior
    prior_label = "uniform"
    shard_dirs: list[str] = []
    cycles: list[dict] = []

    for t in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        shard_dir = out_root / f"cycle_{t:02d}_shards"
        ckpt_dir = out_root / f"cycle_{t:02d}_prior"

        # ---- 1. plan + evaluate + harvest -------------------------------
        print(f"\n{'=' * 70}\nCycle {t}/{args.iterations} — planning with prior: "
              f"{prior_label}\n{'=' * 70}")
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
        pi_log: list = []
        stats = run(
            env, _gumbel_decider(policy, pi_log), args.episodes, args.seed,
            policy=policy,
        )
        if not pi_log:
            raise RuntimeError(
                f"cycle {t} logged no decisions — nothing to distil"
            )
        # One directory per cycle: shard names collide across cycles otherwise.
        _save_pi_targets(pi_log, shard_dir, f"cycle_{t:02d}", args.d_safe)
        shard_dirs.append(str(shard_dir))

        cs = stats["avg_cost_success"]
        print(
            f"  success {stats['success_rate']:.1%}  "
            f"collision {stats['collision_rate']:.1%}  "
            f"cost/success ep {'—' if cs is None else f'{cs:.0f}'}  "
            f"(coarse {stats['avg_coarse_cost']:.0f} + precise "
            f"{stats['avg_precise_cost']:.0f}, precise share "
            f"{stats['precise_cost_share']:.1%})  "
            f"precise {stats['precise_frac']:.1%}  "
            f"transitions/dec {stats['avg_transitions']:.1f}\n"
            f"  breaches: coarse {stats['coarse_breach']}  "
            f"precise {stats['precise_breach']} "
            f"(driven {stats['precise_breach_active']}, "
            f"bystander {stats['precise_breach_bystander']})"
        )

        # ---- 2. distil -> the prior the NEXT cycle plans with ------------
        data_dirs = shard_dirs[-1:] if args.no_replay_mix else shard_dirs
        print(f"\nCycle {t}: distilling {len(data_dirs)} shard dir(s) -> {ckpt_dir}")
        train_stats = train_prior(
            data_dirs, ckpt_dir,
            epochs=args.prior_epochs, lr=args.prior_lr,
            hidden=args.prior_hidden, w_feas=args.w_feas,
            seed=args.seed, device=device, log_every=max(args.prior_epochs // 4, 1),
        )

        cycles.append({
            "name": f"cyc{t}" + ("" if t > 1 else " (unif)"),
            "cycle": t,
            "prior": prior_label,
            "minutes": (time.perf_counter() - t0) / 60.0,
            **stats,
            **train_stats,
        })

        # Next cycle plans with what we just trained.
        prior = LearnedPrior(
            PriorNet.load(train_stats["best_path"], map_location=device),
            feature_builder,
            feas_margin=args.feas_margin,
            device=device,
        )
        prior_label = f"cycle_{t:02d}"

    print_table(cycles, leaf_desc)
    print(
        f"Final prior (trained in cycle {args.iterations}, not yet evaluated): "
        f"{cycles[-1]['best_path']}"
    )

    results_path = out_root / "results.json"
    results_path.write_text(json.dumps(
        {
            "config": vars(args),
            "leaf": leaf_desc,
            "num_robots": sim.num_robots,
            "precise_unit": cost.precise_unit,
            "cycles": cycles,
        },
        indent=2, default=str,
    ))
    print(f"Wrote {results_path}\n")


if __name__ == "__main__":
    main()
