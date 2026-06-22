"""
Ablation harness for the shielded switcher policies P0 and P1.

Runs each rule-based policy (no learning) over a matched set of episodes on top
of the hard safety shield and reports the metrics that answer "is there a
sequential efficiency trade-off worth RL?":

  * success / collision / timeout rates,
  * coarse vs precise usage and a control-cost proxy,
  * shield availability (fraction of decisions with >=1 safe coarse group),
  * shield integrity (collisions that occurred *during* a coarse decision — these
    should be ~0; any non-zero count means d_safe / geometry is miscalibrated).

Read the comparison as:
  P0 ~= P1            -> the steerability/progress axis does not matter; ship P0.
  P1 better than P0   -> "coarse only when productive" helps (build P2 = RL next).

Usage (run on the GPU box — local irsim step crashes; see project memory):
    python -m robot_nav.eval_shield --episodes 100
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
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv
from robot_nav.models.MARL.capswitcher.rl.shielded_policy import ShieldedSwitcher

logger.disable("irsim")

COARSE, PRECISE = 0, 1


def build_env(device: torch.device) -> tuple[SwitcherEnv, CoarseSteering, MARL_SIM_OBSTACLE]:
    """Construct sim + backbone + coarse + env to match the training script."""
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
            "obstacle_6robots_v4/TD3-MARL-obstacle-6robots-reward6"
        ),
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        device=device,
        embedding_source="decoder",
    )
    coarse = CoarseSteering(
        num_robots=sim.num_robots,
        move_distance=1.5,
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
        device=device,
    )
    return env, coarse, sim


def run_policy(
    env: SwitcherEnv,
    policy: ShieldedSwitcher,
    episodes: int,
    base_seed: int,
    c_coarse: float,
    c_precise: float,
) -> dict:
    """Run ``policy`` for ``episodes`` paired (seeded) episodes, collect stats."""
    n = {"success": 0, "collision": 0, "timeout": 0, "oob": 0}
    coarse_dec = precise_dec = total_dec = 0
    safe_available = coarse_breach = 0
    costs, lengths = [], []

    for ep in range(episodes):
        seed = base_seed + ep
        random.seed(seed)
        np.random.seed(seed)
        env.coarse.rng = np.random.default_rng(seed)

        obs = env.reset()  # noqa: F841 (shield reads env._robot_state directly)
        done = False
        ep_cost, ep_len = 0.0, 0
        while not done:
            decision = policy.decide(env._robot_state)
            if any(c.safe for c in decision["candidates"]):
                safe_available += 1

            _, _, done, info = env.step(
                decision["mode"], group=decision["group"], frames=decision["frames"]
            )
            ep_len += 1
            total_dec += 1
            if decision["mode"] == COARSE:
                coarse_dec += 1
                ep_cost += c_coarse
                if info["collision"]:
                    coarse_breach += 1  # shield should make this impossible
            else:
                precise_dec += 1
                ep_cost += c_precise

        if info.get("all_reached"):
            n["success"] += 1
        if info.get("collision"):
            n["collision"] += 1
        if info.get("timeout"):
            n["timeout"] += 1
        if info.get("oob"):
            n["oob"] += 1
        costs.append(ep_cost)
        lengths.append(ep_len)

    return {
        "episodes": episodes,
        "success_rate": n["success"] / episodes,
        "collision_rate": n["collision"] / episodes,
        "timeout_rate": n["timeout"] / episodes,
        "oob_rate": n["oob"] / episodes,
        "avg_decisions": float(np.mean(lengths)),
        "avg_cost": float(np.mean(costs)),
        "coarse_frac": coarse_dec / max(total_dec, 1),
        "coarse_dec": coarse_dec,
        "precise_dec": precise_dec,
        "safe_avail_frac": safe_available / max(total_dec, 1),
        "coarse_breach": coarse_breach,
    }


def print_table(results: dict[str, dict]) -> None:
    """Print a side-by-side comparison of the policy result dicts."""
    rows = [
        ("success rate",        "success_rate",    "{:.1%}"),
        ("collision rate",      "collision_rate",  "{:.1%}"),
        ("timeout rate",        "timeout_rate",    "{:.1%}"),
        ("oob rate",            "oob_rate",         "{:.1%}"),
        ("avg decisions/ep",    "avg_decisions",   "{:.1f}"),
        ("avg control cost/ep", "avg_cost",        "{:.1f}"),
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
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--progress-threshold", type=float, default=0.05)
    ap.add_argument("--c-coarse", type=float, default=1.0)
    ap.add_argument("--c-precise", type=float, default=6.0)
    args = ap.parse_args()

    device = torch.device("cpu")
    env, coarse, sim = build_env(device)
    print(
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"move_distance={coarse.move_distance}, d_safe={args.d_safe}"
    )

    results = {}
    for mode in ("P0", "P1"):
        policy = ShieldedSwitcher(
            coarse, sim, mode=mode,
            d_safe=args.d_safe, progress_threshold=args.progress_threshold,
        )
        print(f"\nRunning {mode} for {args.episodes} episodes ...")
        results[mode] = run_policy(
            env, policy, args.episodes, args.seed, args.c_coarse, args.c_precise
        )

    print_table(results)
    if any(r["coarse_breach"] > 0 for r in results.values()):
        print("WARNING: coarse breaches > 0 — raise --d-safe or check geometry.\n")


if __name__ == "__main__":
    main()
