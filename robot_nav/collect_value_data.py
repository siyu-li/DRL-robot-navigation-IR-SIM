"""
Collect per-robot precise cost-to-go training data (CAPSwitcher value step 1).

Multi-robot scheme: each episode the 6 robots are randomly split into 3 STATIC
robots (action ``[0, 0]`` every sub-step — they act as fixed neighbors /
obstacles) and 3 CONTROLLED robots driven by the frozen precise GAT policy
toward their goals.  The controlled robots are resolved exactly like a
precise-all decision restricted to the controlled set: one DECISION = each
controlled robot in turn is driven by its (per-sub-step recomputed) frozen GAT
action for ``selection_interval`` sub-steps while every other robot holds still
— mirroring :meth:`SwitcherEnv._run_precise`.

Per controlled robot, per decision boundary, we log both feature variants:

    emb  = frozen GAT per-robot embedding e_i (512-d) — because the 3 static
           robots are present, e_i carries real robot-neighbor attention;
    geo  = the 11-col robot state row [px,py,cos,sin,dist,cos_err,sin_err,
           lin,ang,gx,gy];

and backfill the label at episode end:

    label_i(t) = (R_i + 1 - t) * precise_cost

where ``R_i`` is the (0-based) decision during which robot i first came within
``goal_threshold`` of its goal.  This is the Monte-Carlo realized remaining
precise cost in PathCostReward units: the reward charges a flat
``precise_cost`` per precise decision, and a robot receives exactly
``selection_interval`` GAT sub-steps per decision both here (3 controlled
robots take turns) and in precise-all (6 robots take turns), so per-robot
decision counts transfer.  Labeling stops at first arrival (cost-to-go 0 at
goal); robots that never arrive (collision / decision cap) are CENSORED and
discarded — the other controlled robots of the same episode are kept.

Caveat carried by the data: the neighbors are STATIC, so the label is a
cost-to-go treating them as fixed — a partial-congestion model, not
moving-agent congestion.

Run on the GPU box (local irsim step crashes; see project memory):
    python -m robot_nav.collect_value_data --episodes 1000 --out data/value_data
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.eval_mpc import build_env
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv

logger.disable("irsim")


def _drive_decision(
    env: SwitcherEnv,
    controlled: list[int],
    reach_dec: dict[int, int | None],
    dec: int,
    goal_threshold: float,
) -> tuple[bool, dict]:
    """
    Execute one collection decision: each controlled robot in turn gets
    ``selection_interval`` GAT-driven sub-steps while every other robot holds
    still (static robots and waiting controlled robots receive ``[0, 0]``).

    First goal arrivals are recorded into ``reach_dec`` after every sub-step.

    Returns:
        (done, info): sim-terminal flag (collision / all-goals) and the info
        dict populated by ``_apply_substep``.
    """
    info: dict = {"collision": False, "all_reached": False, "oob": False}
    n = env.sim.num_robots
    for r in controlled:
        for _ in range(env.selection_interval):
            if not env._cache_valid:
                env._refresh_cache()
            raw = env._cached_raw_actions
            # Only robot r moves; actor-space -> sim-input:
            # lin = (raw[0]+1)/4 in [0, 0.5], ang = raw[1].
            sim_actions = [
                [(float(raw[i, 0]) + 1.0) / 4.0, float(raw[i, 1])]
                if i == r
                else [0.0, 0.0]
                for i in range(n)
            ]
            done = env._apply_substep(sim_actions, info)
            env._cache_valid = False
            for i in controlled:
                if reach_dec[i] is None and env._last_distances[i] <= goal_threshold:
                    reach_dec[i] = dec
            if done:
                return True, info
    return False, info


def collect_episode(
    env: SwitcherEnv,
    split_rng: np.random.Generator,
    max_decisions: int,
    goal_threshold: float,
    precise_cost: float,
) -> tuple[list[tuple], dict]:
    """
    Run one collection episode and return its labeled samples.

    Returns:
        rows:  list of (emb(512,), geo(11,), label, robot_id, decision_idx) for
               every kept (non-censored) sample.
        stats: {"reached", "censored", "collision"} for aggregate reporting.
    """
    env.reset()
    n = env.sim.num_robots
    controlled = sorted(int(i) for i in split_rng.choice(n, size=n // 2, replace=False))

    reach_dec: dict[int, int | None] = {i: None for i in controlled}
    for i in controlled:  # already at goal before any decision: no samples
        if env._last_distances[i] <= goal_threshold:
            reach_dec[i] = -1

    samples: dict[int, list] = {i: [] for i in controlled}
    collision = False
    for dec in range(max_decisions):
        pending = [i for i in controlled if reach_dec[i] is None]
        if not pending:
            break
        h = env._get_obs()          # (N, 512) embeddings of the boundary state
        rs = env._robot_state       # (N, 11)
        for i in pending:
            samples[i].append(
                (np.array(h[i], dtype=np.float32), np.array(rs[i], dtype=np.float32), dec)
            )
        done, info = _drive_decision(env, controlled, reach_dec, dec, goal_threshold)
        if done:
            collision = bool(info["collision"])
            break

    rows: list[tuple] = []
    reached = censored = 0
    for i in controlled:
        if reach_dec[i] is None:            # censored: never arrived — discard
            censored += 1
            continue
        reached += 1
        for emb, geo, t in samples[i]:      # reach_dec == -1 has no samples
            label = (reach_dec[i] + 1 - t) * precise_cost
            rows.append((emb, geo, float(label), i, t))
    return rows, {"reached": reached, "censored": censored, "collision": int(collision)}


def _save_shard(out_dir: Path, shard_idx: int, rows: list[tuple], episodes: list[int], meta: dict) -> None:
    """Write one .npz shard of labeled samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"value_data_{shard_idx:05d}.npz"
    np.savez_compressed(
        path,
        emb=np.stack([r[0] for r in rows]).astype(np.float32),
        geo=np.stack([r[1] for r in rows]).astype(np.float32),
        label=np.array([r[2] for r in rows], dtype=np.float32),
        robot=np.array([r[3] for r in rows], dtype=np.int16),
        decision=np.array([r[4] for r in rows], dtype=np.int16),
        episode=np.array(episodes, dtype=np.int32),
        **{f"meta_{k}": v for k, v in meta.items()},
    )
    print(f"  saved {len(rows)} samples -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--out", type=str, default="data/value_data")
    ap.add_argument("--max-decisions", type=int, default=60)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--shard-episodes", type=int, default=200,
                    help="episodes per .npz shard")
    ap.add_argument("--move-distance", type=float, default=1.5,
                    help="only affects env construction; coarse mode is unused here")
    args = ap.parse_args()

    device = torch.device("cpu")
    env, coarse, sim = build_env(device, move_distance=args.move_distance)
    precise_cost = float(env.reward_fn.precise_cost)
    meta = {
        "precise_cost": precise_cost,
        "selection_interval": env.selection_interval,
        "goal_threshold": args.goal_threshold,
        "step_time": float(sim.env.step_time),
        "lin_max": float(coarse.lin_max),
        "num_robots": sim.num_robots,
    }
    print(
        f"Env: {sim.num_robots} robots ({sim.num_robots // 2} controlled / "
        f"{sim.num_robots - sim.num_robots // 2} static per episode), "
        f"precise_cost={precise_cost}, selection_interval={env.selection_interval}, "
        f"max_decisions={args.max_decisions}"
    )

    out_dir = Path(args.out)
    shard_rows: list[tuple] = []
    shard_eps: list[int] = []
    shard_idx = 0
    total = {"samples": 0, "reached": 0, "censored": 0, "collision": 0}

    for ep in range(args.episodes):
        seed = args.seed + ep
        random.seed(seed)
        np.random.seed(seed)
        split_rng = np.random.default_rng(seed)

        rows, stats = collect_episode(
            env, split_rng, args.max_decisions, args.goal_threshold, precise_cost
        )
        shard_rows.extend(rows)
        shard_eps.extend([ep] * len(rows))
        total["samples"] += len(rows)
        total["reached"] += stats["reached"]
        total["censored"] += stats["censored"]
        total["collision"] += stats["collision"]

        if (ep + 1) % 50 == 0:
            print(
                f"[{ep + 1}/{args.episodes}] samples={total['samples']} "
                f"reached={total['reached']} censored={total['censored']} "
                f"collision_eps={total['collision']}"
            )
        if (ep + 1) % args.shard_episodes == 0 and shard_rows:
            _save_shard(out_dir, shard_idx, shard_rows, shard_eps, meta)
            shard_idx += 1
            shard_rows, shard_eps = [], []

    if shard_rows:
        _save_shard(out_dir, shard_idx, shard_rows, shard_eps, meta)

    n_traj = total["reached"] + total["censored"]
    print(
        f"\nDone: {total['samples']} samples from {args.episodes} episodes; "
        f"controlled-robot success {total['reached']}/{n_traj} "
        f"({total['reached'] / max(n_traj, 1):.1%}), "
        f"collision episodes {total['collision']}."
    )


if __name__ == "__main__":
    main()
