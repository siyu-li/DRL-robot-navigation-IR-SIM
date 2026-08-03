"""
Model-vs-sim fidelity check for the MPC analytic forward model (Task 6, blocking).

The depth-d lookahead runs on the analytic pose-model in ``rl/forward_model.py``
instead of the (un-branchable) simulator.  This script verifies the model's one-step
predictions against the real ``SwitcherEnv.step`` so the planner ranks actions on
faithful transitions:

  * **Coarse** should match *exactly* — both the model and the env execute the same
    vetted frames, and both reconstruct straight-line member motion (shield
    ``swept_positions``).  Any mismatch is a bug.
  * **Precise** should match *within tolerance* — the model integrates unicycle
    dynamics (forward Euler) as a stand-in for irsim.  If the error is large, irsim
    likely integrates θ before translating; switch ``ForwardModel._integrate`` to a
    semi-implicit update and re-check.

Run on the GPU box (local irsim step crashes; see project memory):
    python -m robot_nav.check_mpc_model --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from robot_nav.eval_mpc import DEFAULT_COST_CONFIG, build_env
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.forward_model import (
    ForwardModel,
    _state_group_seed,
)
from robot_nav.models.MARL.capswitcher.rl.search.minimin import MPCSwitcher


def _sim_poses(env) -> np.ndarray:
    """Current per-robot poses [x, y, theta] from the live sim."""
    return np.asarray(env._poses, dtype=np.float64)[:, :3]


def _pose_err(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Max XY error (m) and max wrapped-heading error (rad)."""
    dxy = float(np.linalg.norm(pred[:, :2] - actual[:, :2], axis=1).max())
    dth = np.abs((pred[:, 2] - actual[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    return dxy, float(dth.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG,
                    help="SwitcherCost YAML")
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    args = ap.parse_args()

    device = torch.device("cpu")
    cost = SwitcherCost.from_yaml(args.cost_config)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold
    )
    switcher = MPCSwitcher(env.backbone, coarse, sim, depth=1, cost=cost)

    coarse_xy = coarse_th = 0.0
    precise_xy = precise_th = 0.0
    n_coarse = n_precise = 0

    # Drive with a SAFE policy — coarse only when the shield vets a group safe,
    # else precise — so executed steps don't collide (a collision ends the episode
    # before anything is measured).  Alternate the preference so both modes get
    # sampled.  For each executed step, predict end-poses from the model at the
    # decision state, execute the *same* action, and compare.
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        env.coarse.rng = np.random.default_rng(args.seed + ep)
        env.reset()

        prefer_coarse = True
        for _ in range(40):
            rs = env._robot_state
            model = switcher._build_model(rs)
            ms = ForwardModel.state_from_robot_state(rs)

            safe = [m for m in model.coarse_moves(ms).values() if m.candidate.safe]
            use_coarse = prefer_coarse and len(safe) > 0
            prefer_coarse = not prefer_coarse

            if use_coarse:
                move = safe[0]
                pred = move.next_state.poses.copy()
                _, _, done, info = env.step(
                    0, group=move.candidate.group, frames=move.candidate.frames
                )
                if not (info["collision"] or info["oob"]):
                    dxy, dth = _pose_err(pred, _sim_poses(env))
                    coarse_xy = max(coarse_xy, dxy); coarse_th = max(coarse_th, dth)
                    n_coarse += 1
            else:
                pred = model.precise_next(ms).poses.copy()
                _, _, done, info = env.step(1)
                if not (info["collision"] or info["oob"]):
                    dxy, dth = _pose_err(pred, _sim_poses(env))
                    precise_xy = max(precise_xy, dxy); precise_th = max(precise_th, dth)
                    n_precise += 1
            if done:
                break

    print("\n=== MPC model-vs-sim fidelity ===")
    print(f"coarse  (n={n_coarse:3d}):  max XY err = {coarse_xy:.2e} m   max θ err = {coarse_th:.2e} rad")
    print(f"precise (n={n_precise:3d}):  max XY err = {precise_xy:.2e} m   max θ err = {precise_th:.2e} rad")
    print("\nExpect coarse ~0 (exact). Precise small; if large, try a semi-implicit "
          "integrator in ForwardModel._integrate.\n")


if __name__ == "__main__":
    main()
