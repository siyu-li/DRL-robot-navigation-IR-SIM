"""
Batch evaluation script for CAPSwitcher — runs N headless episodes and reports
collision rate, success rate, and average steps per episode.

Usage
-----
    python -m robot_nav.scripts.eval_capswitcher --mode switcher --num-runs 50
    python -m robot_nav.scripts.eval_capswitcher --mode coarse   --num-runs 50
    python -m robot_nav.scripts.eval_capswitcher --mode precise  --num-runs 50

Modes
-----
switcher  Load the trained DQN checkpoint and select mode greedily (ε = 0).
coarse    Always choose action 0 — every decision is a coarse group control.
precise   Always choose action 1 — every decision is precise sequential GAT.

Metrics
-------
- Success rate    : fraction of episodes where all robots reached their goals.
- Collision rate  : fraction of episodes that ended in a robot collision.
- Timeout rate    : fraction of episodes that exhausted the decision budget.
- OOB events      : total number of sub-steps where any robot left world bounds
                    (OOB no longer terminates an episode — robots keep running).
- Avg sim steps   : mean total simulator sub-steps per episode.
- Avg decisions   : mean number of switcher decisions taken per episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.rl.switcher_dqn import DeepSetsQNet
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv

logger.disable("irsim")

# Default checkpoint — the final trained policy
_DEFAULT_CKPT = Path(
    "robot_nav/models/MARL/capswitcher/checkpoints/cap_switcher/"
    "capswitcher_dqn_final.pth"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluation for CAPSwitcher (plotting disabled)."
    )
    p.add_argument(
        "--mode",
        choices=["switcher", "coarse", "precise"],
        default="switcher",
        help="Control mode to evaluate (default: switcher)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CKPT,
        help=f"DQN checkpoint .pth used in switcher mode (default: {_DEFAULT_CKPT})",
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of evaluation episodes to run (default: 50)",
    )
    p.add_argument(
        "--max-decisions",
        type=int,
        default=80,
        help="Episode budget in switcher decisions (default: 80)",
    )
    p.add_argument(
        "--selection-interval",
        type=int,
        default=5,
        help="Precise-mode sub-steps per robot per decision (default: 5)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  CAPSwitcher evaluation  |  mode: {args.mode.upper()}")
    print(f"  Runs: {args.num_runs}  |  Max decisions/episode: {args.max_decisions}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Simulation (plotting DISABLED for batch evaluation)                 #
    # ------------------------------------------------------------------ #
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle.yaml",
        disable_plotting=True,
        reward_phase=6,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=1.5,
        num_inactive_robots=0,
    )
    print(
        f"Environment: {sim.num_robots} robots, "
        f"{sim.num_obstacles} obstacles, "
        f"world x={sim.x_range} y={sim.y_range}\n"
    )

    # ------------------------------------------------------------------ #
    # Frozen GAT backbone                                                  #
    # ------------------------------------------------------------------ #
    gat_backbone = GATBackbone(
        # checkpoint_path=Path(
        #     "robot_nav/models/MARL/marlTD3/checkpoint/"
        #     "Mar.15_obstacle_14robot_reward8/TD3-MARL-obstacle-14robots",
        # ),
        checkpoint_path=Path(
            "robot_nav/models/MARL/marlTD3/checkpoint/"
            "Mar.04_obstacle_14robots_partial_inactive/TD3-MARL-obstacle-14robots-partial-inactive_epoch210"
        ),
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        device=device,
        embedding_source="decoder",
    )

    # ------------------------------------------------------------------ #
    # Coarse steering                                                      #
    # ------------------------------------------------------------------ #
    coarse_steering = CoarseSteering(
        num_robots=sim.num_robots,
        move_distance=1.5,
        method="nonlinear",
        step_time=sim.env.step_time,
        ang_max=1.0,
        lin_max=0.5,
    )

    # ------------------------------------------------------------------ #
    # Switcher environment                                                 #
    # ------------------------------------------------------------------ #
    env = SwitcherEnv(
        sim=sim,
        backbone=gat_backbone,
        coarse_steering=coarse_steering,
        selection_interval=args.selection_interval,
        max_decisions=args.max_decisions,
        device=device,
        terminate_on_oob=False,      # OOB is logged but does NOT end the episode
    )

    # ------------------------------------------------------------------ #
    # Q-network (only loaded for switcher mode)                           #
    # ------------------------------------------------------------------ #
    q_net: DeepSetsQNet | None = None
    if args.mode == "switcher":
        q_net = DeepSetsQNet(
            embed_dim=env.EMBED_DIM,
            phi_dims=(256, 128),
            rho_dims=(128,),
            num_actions=env.NUM_ACTIONS,
            aggregation="sum_max",
        ).to(device)
        state_dict = torch.load(str(args.checkpoint), map_location=device)
        q_net.load_state_dict(state_dict)
        q_net.eval()
        print(f"Loaded DQN checkpoint: {args.checkpoint}\n")

    # ------------------------------------------------------------------ #
    # Per-episode trackers                                                 #
    # ------------------------------------------------------------------ #
    n_success   = 0
    n_collision = 0
    n_timeout   = 0
    n_oob_episodes = 0   # episodes that had at least one OOB event

    total_sim_steps  = 0   # sum of env._step_count across all episodes
    total_decisions  = 0   # sum of env._decision_count across all episodes

    coarse_decisions_total  = 0
    precise_decisions_total = 0

    # ------------------------------------------------------------------ #
    # Per-run table header                                                 #
    # ------------------------------------------------------------------ #
    print(
        f"  {'Run':>5}  {'Result':>10}  {'SimSteps':>9}  "
        f"{'Decisions':>9}  {'Coarse%':>8}"
    )
    print(f"  {'-'*52}")

    for run in range(1, args.num_runs + 1):
        obs  = env.reset()
        done = False

        run_coarse  = 0
        run_precise = 0

        while not done:
            # ---- Action selection ---------------------------------------
            if args.mode == "switcher":
                with torch.no_grad():
                    obs_t  = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)            # (1, N, 512)
                    action = int(q_net(obs_t).argmax(dim=1).item())
            elif args.mode == "coarse":
                action = 0
            else:   # precise
                action = 1

            obs, reward, done, info = env.step(action)

            if action == 0:
                run_coarse  += 1
            else:
                run_precise += 1

        # ---- Accumulate stats -------------------------------------------
        sim_steps = env._step_count
        decisions = env._decision_count

        total_sim_steps += sim_steps
        total_decisions += decisions
        coarse_decisions_total  += run_coarse
        precise_decisions_total += run_precise

        if info["all_reached"]:
            n_success += 1
            result = "SUCCESS"
        elif info["collision"]:
            n_collision += 1
            result = "COLLISION"
        else:   # timeout
            n_timeout += 1
            result = "TIMEOUT"

        if info.get("oob"):
            n_oob_episodes += 1

        coarse_pct = 100.0 * run_coarse / decisions if decisions > 0 else 0.0
        print(
            f"  {run:>5d}  {result:>10}  {sim_steps:>9d}  "
            f"{decisions:>9d}  {coarse_pct:>7.1f}%"
        )

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    N = args.num_runs
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — {args.mode.upper()} mode ({N} runs)")
    print(f"{'='*60}")
    print(f"  Success   rate : {n_success  / N * 100:6.1f}%  ({n_success}/{N})")
    print(f"  Collision rate : {n_collision / N * 100:6.1f}%  ({n_collision}/{N})")
    print(f"  Timeout   rate : {n_timeout   / N * 100:6.1f}%  ({n_timeout}/{N})")
    print(f"  OOB events     : {n_oob_episodes} episode(s) had at least one OOB sub-step (non-terminal)")
    print(f"  Avg sim steps  : {total_sim_steps / N:8.1f}")
    print(f"  Avg decisions  : {total_decisions / N:8.1f}")
    if args.mode == "switcher":
        total_dec = coarse_decisions_total + precise_decisions_total
        if total_dec > 0:
            print(
                f"  Coarse share   : {coarse_decisions_total  / total_dec * 100:6.1f}%"
                f"  Precise share  : {precise_decisions_total / total_dec * 100:6.1f}%"
            )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
