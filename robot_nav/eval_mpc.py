"""
Evaluation harness for the fixed-depth receding-horizon MPC switcher.

Runs the MPC switcher at several lookahead depths against the two mode-only
baselines (precise-only, coarse-only) over a matched set of seeded episodes on
6 robots / 3 coarse groups, and reports the metrics that answer "does sequential
lookahead help?":

  * success / collision / timeout rates,
  * coarse vs precise usage and travel distance (path length, all robots),
  * shield availability (fraction of decisions with >=1 safe coarse group),
  * shield integrity (collisions *during* a coarse decision — should be ~0).

Decision gate: if MPC-d2/d3 does not beat MPC-d1 on precise fraction at equal
success, sequential lookahead adds nothing here (stop before AlphaZero).

Usage (run on the GPU box — local irsim step crashes; see project memory):
    python -m robot_nav.eval_mpc --episodes 100 --depths 1 2 3
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.rl.reward import PathCostReward
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv
from robot_nav.models.MARL.capswitcher.rl.mpc.mpc_switcher import MPCSwitcher

logger.disable("irsim")

COARSE, PRECISE = 0, 1


def build_env(
    device: torch.device, move_distance: float
) -> tuple[SwitcherEnv, CoarseSteering, MARL_SIM_OBSTACLE]:
    """Construct sim + backbone + coarse + env (mirrors eval_shield.build_env)."""
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle.yaml",
        disable_plotting=True,
        reward_phase=6,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=1.5,
        num_inactive_robots=0,
    )
    backbone = GATBackbone(
        checkpoint_path=Path(
            "robot_nav/models/MARL/marlTD3/checkpoint/"
            "Mar.04_obstacle_14robots_partial_inactive/"
            "TD3-MARL-obstacle-14robots-partial-inactive_epoch210"
        ),
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        device=device,
        embedding_source="decoder",
    )
    coarse = CoarseSteering(
        num_robots=sim.num_robots,
        move_distance=move_distance,
        method="nonlinear",
        step_time=sim.env.step_time,
        ang_max=1.0,
        lin_max=0.5,
    )
    env = SwitcherEnv(
        sim=sim,
        backbone=backbone,
        coarse_steering=coarse,
        selection_interval=5,
        max_decisions=60,
        reward_fn=PathCostReward(),
        device=device,
        terminate_on_oob=False,
    )
    return env, coarse, sim


def run(env: SwitcherEnv, decide_fn, episodes: int, base_seed: int) -> dict:
    """
    Run ``decide_fn`` for ``episodes`` seeded episodes and collect stats.

    ``decide_fn(env) -> {"mode", "group", "frames", "candidates"}``.  Per-episode
    cost is travel distance (``info["path_cost"]``), summed over robots and the
    episode — the same quantity ``PathCostReward`` penalises during training.
    """
    n = {"success": 0, "collision": 0, "timeout": 0}
    coarse_dec = precise_dec = total_dec = 0
    safe_available = coarse_breach = 0
    costs, lengths = [], []

    for ep in range(episodes):
        seed = base_seed + ep
        random.seed(seed)
        np.random.seed(seed)
        env.coarse.rng = np.random.default_rng(seed)
        env._coarse_rng = np.random.default_rng(seed)  # coarse-only group choice

        env.reset()
        done = False
        ep_cost, ep_len = 0.0, 0
        info: dict = {}
        while not done:
            decision = decide_fn(env)
            cands = decision.get("candidates") or []
            if any(getattr(c, "safe", False) for c in cands):
                safe_available += 1

            _, _, done, info = env.step(
                decision["mode"], group=decision["group"], frames=decision["frames"]
            )
            ep_len += 1
            total_dec += 1
            ep_cost += float(info["path_cost"])
            if decision["mode"] == COARSE:
                coarse_dec += 1
                if info["collision"]:
                    coarse_breach += 1
            else:
                precise_dec += 1

        if info.get("all_reached"):
            n["success"] += 1
        if info.get("collision"):
            n["collision"] += 1
        if info.get("timeout"):
            n["timeout"] += 1
        costs.append(ep_cost)
        lengths.append(ep_len)

    return {
        "episodes": episodes,
        "success_rate": n["success"] / episodes,
        "collision_rate": n["collision"] / episodes,
        "timeout_rate": n["timeout"] / episodes,
        "avg_decisions": float(np.mean(lengths)),
        "avg_cost": float(np.mean(costs)),
        "coarse_frac": coarse_dec / max(total_dec, 1),
        "precise_frac": precise_dec / max(total_dec, 1),
        "safe_avail_frac": safe_available / max(total_dec, 1),
        "coarse_breach": coarse_breach,
    }


# ---- decision sources -----------------------------------------------------

def _precise_only(env: SwitcherEnv) -> dict:
    return {"mode": PRECISE, "group": None, "frames": None, "candidates": []}


def _coarse_only(env: SwitcherEnv) -> dict:
    # group=None → SwitcherEnv picks a uniform-random selectable group.
    return {"mode": COARSE, "group": None, "frames": None, "candidates": []}


def _mpc_decider(policy: MPCSwitcher):
    def decide(env: SwitcherEnv) -> dict:
        return policy.decide(env._robot_state)

    return decide


def print_table(results: dict[str, dict]) -> None:
    """Side-by-side comparison of the result dicts."""
    rows = [
        ("success rate",        "success_rate",    "{:.1%}"),
        ("collision rate",      "collision_rate",  "{:.1%}"),
        ("timeout rate",        "timeout_rate",    "{:.1%}"),
        ("avg decisions/ep",    "avg_decisions",   "{:.1f}"),
        ("avg path length/ep",  "avg_cost",        "{:.1f}"),
        ("precise fraction",    "precise_frac",    "{:.1%}"),
        ("coarse fraction",     "coarse_frac",     "{:.1%}"),
        ("safe-coarse avail.",  "safe_avail_frac", "{:.1%}"),
        ("coarse breaches (!)", "coarse_breach",   "{:d}"),
    ]
    names = list(results)
    header = f"{'metric':<22}" + "".join(f"{nm:>14}" for nm in names)
    print("\n" + header)
    print("-" * len(header))
    for label, key, fmt in rows:
        line = f"{label:<22}"
        for nm in names:
            line += f"{fmt.format(results[nm][key]):>14}"
        print(line)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=None,
                    help="cost-to-go slope; default precise_cost/(lin_max·step_time)")
    ap.add_argument("--move-distance", type=float, default=1.5)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--baselines", action="store_true",
                    help="also run precise-only and coarse-only baselines")
    ap.add_argument("--value-model", type=str, default=None,
                    help="learned cost-to-go checkpoint (train_value.py); when set, "
                         "each depth also runs with the learned leaf (rows MPC-d*+v)")
    args = ap.parse_args()

    device = torch.device("cpu")
    env, coarse, sim = build_env(device, move_distance=args.move_distance)
    print(
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"move_distance={coarse.move_distance}, d_safe={args.d_safe}, "
        f"alpha={args.alpha}, depths={args.depths}, value_model={args.value_model}"
    )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        print(f"Learned leaf: feature={leaf_value.feature}, "
              f"precise_cost={leaf_value.precise_cost}")

    results: dict[str, dict] = {}

    if args.baselines:
        print(f"\nRunning precise-only for {args.episodes} episodes ...")
        results["precise"] = run(env, _precise_only, args.episodes, args.seed)
        print(f"Running coarse-only for {args.episodes} episodes ...")
        results["coarse"] = run(env, _coarse_only, args.episodes, args.seed)

    for d in args.depths:
        variants = [("", None)] + ([("+v", leaf_value)] if leaf_value else [])
        for tag, leaf in variants:
            policy = MPCSwitcher(
                backbone=env.backbone,
                coarse=coarse,
                sim=sim,
                depth=d,
                d_safe=args.d_safe,
                alpha=args.alpha,
                selection_interval=env.selection_interval,
                goal_threshold=args.goal_threshold,
                reward_fn=env.reward_fn,
                leaf_value=leaf,
            )
            name = f"MPC-d{d}{tag}"
            print(f"\nRunning {name} for {args.episodes} episodes ...")
            results[name] = run(env, _mpc_decider(policy), args.episodes, args.seed)

    print_table(results)
    if any(r["coarse_breach"] > 0 for r in results.values()):
        print("WARNING: coarse breaches > 0 — raise --d-safe or check geometry.\n")


if __name__ == "__main__":
    main()
