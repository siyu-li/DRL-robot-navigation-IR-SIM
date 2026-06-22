"""
Visual test script for CAPSwitcher — renders one episode with a chosen mode.

Usage
-----
    python -m robot_nav.scripts.test_capswitcher --mode switcher
    python -m robot_nav.scripts.test_capswitcher --mode coarse
    python -m robot_nav.scripts.test_capswitcher --mode precise

Modes
-----
switcher  Load the trained DQN checkpoint and select mode greedily (ε = 0).
coarse    Always choose action 0 — every decision is a coarse group control.
precise   Always choose action 1 — every decision is precise sequential GAT.

Plotting is enabled so you can watch the simulation render live.
No statistics are computed; print output shows every decision.
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
        description="Visual episode test for CAPSwitcher (rendering enabled)."
    )
    p.add_argument(
        "--mode",
        choices=["switcher", "coarse", "precise"],
        default="switcher",
        help="Control mode to run (default: switcher)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CKPT,
        help=f"DQN checkpoint .pth used in switcher mode (default: {_DEFAULT_CKPT})",
    )
    p.add_argument(
        "--max-decisions",
        type=int,
        default=60,
        help="Episode budget in switcher decisions (default: 60)",
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
    print(f"  CAPSwitcher visual test  |  mode: {args.mode.upper()}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Simulation (plotting ON for visual test)                            #
    # ------------------------------------------------------------------ #
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle.yaml",
        disable_plotting=False,          # <-- live rendering
        reward_phase=6,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=1.5,
        num_inactive_robots=0,
    )
    print(
        f"Environment: {sim.num_robots} robots, "
        f"{sim.num_obstacles} obstacles, "
        f"world x={sim.x_range} y={sim.y_range}"
    )

    # ------------------------------------------------------------------ #
    # Frozen GAT backbone                                                  #
    # ------------------------------------------------------------------ #
    gat_backbone = GATBackbone(
        checkpoint_path=Path(
            "robot_nav/models/MARL/marlTD3/checkpoint/"
            "obstacle_6robots_v4/TD3-MARL-obstacle-6robots-reward6"
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
    )

    # ------------------------------------------------------------------ #
    # Q-network (only loaded for switcher mode)                           #
    # Using DeepSetsQNet directly avoids allocating a large replay buffer #
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
    # Episode rollout                                                      #
    # ------------------------------------------------------------------ #
    obs = env.reset()
    done = False
    decision = 0

    print(f"Running episode (max {args.max_decisions} decisions) ...\n")
    print(
        f"  {'Decision':>8}  {'Mode':>7}  {'Reward':>9}  "
        f"{'SubSteps':>8}  {'Collision':>9}  {'AllReached':>10}"
    )
    print(f"  {'-'*60}")

    while not done:
        decision += 1

        # ---- Action selection -------------------------------------------
        if args.mode == "switcher":
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)   # (1, N, 512)
                action = int(q_net(obs_t).argmax(dim=1).item())
        elif args.mode == "coarse":
            action = 0
        else:   # precise
            action = 1

        obs, reward, done, info = env.step(action)

        mode_label = "COARSE " if action == 0 else "PRECISE"
        print(
            f"  {decision:>8d}  {mode_label}  {reward:>+9.3f}  "
            f"{info['steps_taken']:>8d}  {str(info['collision']):>9}  "
            f"{str(info['all_reached']):>10}"
        )

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  Episode ended after {decision} decisions.")
    if info["all_reached"]:
        print("  Result: ALL GOALS REACHED ✓")
    elif info["collision"]:
        print("  Result: COLLISION ✗")
    elif info.get("timeout"):
        print("  Result: TIMEOUT (decision budget exhausted)")
    elif info.get("oob"):
        print("  Result: OUT OF BOUNDS")
    else:
        print("  Result: (unknown termination)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
