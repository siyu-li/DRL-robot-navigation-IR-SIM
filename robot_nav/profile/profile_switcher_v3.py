"""
Profile the switcher RL training loop (v3) — fine-grained breakdown of env.step().

Manually replicates the inner loop from SwitcherEnv.step() with timing
on each sub-component, matching the actual optimised code paths:
  - Cached obstacle tensor (self._obstacle_t)
  - Combined _get_actions_and_embeddings (attention + policy_head in one pass)
  - Cached embeddings in _build_obs
  - Vectorized _get_extra_features
  - Batch _objects_step in sim.step

Usage:
    python -m robot_nav.profile_switcher_v3
"""

import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.switcher.group_generator import generate_all_groups
from robot_nav.models.MARL.switcher.rl_feature_builder import RLFeatureBuilder
from robot_nav.models.MARL.switcher.config_loader import load_switcher_config
from robot_nav.models.MARL.switcher.switcher_env import SwitcherEnv
from robot_nav.models.MARL.switcher.switcher_ppo import SwitcherPPO
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

from loguru import logger
logger.disable("irsim")


# ── Timer ─────────────────────────────────────────────────────────────
class Timer:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def __call__(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self.totals[name] += elapsed
        self.counts[name] += 1

    def report(self, title: str = "Profile Results"):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        total_all = sum(self.totals.values())
        items = sorted(self.totals.items(), key=lambda x: -x[1])
        print(f"{'Section':<45} {'Total(s)':>8} {'Calls':>7} {'Avg(ms)':>9} {'%':>6}")
        print(f"{'-'*45} {'-'*8} {'-'*7} {'-'*9} {'-'*6}")
        for name, total in items:
            count = self.counts[name]
            avg_ms = (total / count) * 1000 if count > 0 else 0
            pct = (total / total_all * 100) if total_all > 0 else 0
            print(f"{name:<45} {total:8.3f} {count:7d} {avg_ms:9.3f} {pct:5.1f}%")
        print(f"{'='*70}\n")


timer = Timer()


# ── Config ────────────────────────────────────────────────────────────
CONFIG = {
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "n_robots": 14,
    "n_obstacles": 7,
    "disable_plotting": True,
    "obstacle_proximity_threshold": 1.5,
    "goal_threshold": 0.3,
    "max_episode_steps": 1000,
    "selection_interval": 10,
    "reward_phase": 5,
    "per_robot_goal_reset": False,
    "state_dim": 11,
    "obstacle_state_dim": 4,
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-gpu_epoch800",
    "decentralized_model_dir": "robot_nav/models/MARL/marlTD3/checkpoint/"
                               "Feb.10_obstacle_14robot_transfer_gpu",
    "include_sizes": (1, 2, 3, 4, 7),
    "embed_dim": 512,
    "lr_actor": 3e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "eps_clip": 0.2,
    "entropy_coeff": 0.02,
    "value_coeff": 0.5,
    "max_grad_norm": 1.0,
    "ppo_epochs": 10,
    "embed_hidden": 256,
    "group_scalar_hidden": 64,
    "fusion_hidden": 256,
    "value_embed_hidden": 128,
    "value_scalar_hidden": 32,
    "model_name": "SwitcherPPO-14robots-profile",
    "save_directory": "/tmp/switcher_profile",
    "seed": 42,
}

PROFILE_DECISIONS = 100


def _generate_groups(num_robots, include_sizes):
    m = 4 if num_robots > 6 else 3
    all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)
    allowed = set(include_sizes)
    return [g for g in all_groups if len(g) in allowed]


def main():
    cfg = CONFIG
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Setup ──
    sim = MARL_SIM_OBSTACLE(
        world_file=cfg["world_file"],
        disable_plotting=cfg["disable_plotting"],
        reward_phase=cfg["reward_phase"],
        per_robot_goal_reset=cfg["per_robot_goal_reset"],
        obstacle_proximity_threshold=cfg["obstacle_proximity_threshold"],
    )

    policy = TD3Obstacle(
        state_dim=cfg["state_dim"],
        action_dim=2,
        max_action=1,
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        obstacle_state_dim=cfg["obstacle_state_dim"],
        device=device,
        load_model=True,
        model_name=cfg["decentralized_model_name"],
        load_model_name=cfg["decentralized_model_name"],
        load_directory=Path(cfg["decentralized_model_dir"]),
        save_directory=Path(cfg["decentralized_model_dir"]),
        inference_only=True,
    )
    policy.actor.eval()

    groups = _generate_groups(cfg["n_robots"], cfg["include_sizes"])
    print(f"Groups: {len(groups)}")

    sw_cfg = load_switcher_config("robot_nav/models/MARL/switcher/switcher_config.yaml")
    fb = RLFeatureBuilder.from_config(
        sw_cfg,
        embed_dim=cfg["embed_dim"],
        pooling="mean",
        max_group_size=max(len(g) for g in groups),
    )

    env = SwitcherEnv(
        sim=sim,
        policy=policy,
        groups=groups,
        feature_builder=fb,
        selection_interval=cfg["selection_interval"],
        max_episode_steps=cfg["max_episode_steps"],
        goal_threshold=cfg["goal_threshold"],
        device=device_str,
    )

    ppo = SwitcherPPO(
        embed_dim=cfg["embed_dim"],
        group_scalar_dim=sw_cfg.group_scalar_dim,
        state_scalar_dim=sw_cfg.state_scalar_dim,
        lr_actor=cfg["lr_actor"],
        lr_critic=cfg["lr_critic"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        eps_clip=cfg["eps_clip"],
        entropy_coeff=cfg["entropy_coeff"],
        value_coeff=cfg["value_coeff"],
        max_grad_norm=cfg["max_grad_norm"],
        device=device_str,
        save_every=9999,
        model_name=cfg["model_name"],
        save_directory=Path(cfg["save_directory"]),
        embed_hidden=cfg["embed_hidden"],
        group_scalar_hidden=cfg["group_scalar_hidden"],
        fusion_hidden=cfg["fusion_hidden"],
        value_embed_hidden=cfg["value_embed_hidden"],
        value_scalar_hidden=cfg["value_scalar_hidden"],
    )

    # ── Warm-up ──
    gf, sf = env.reset()
    idx = ppo.get_action(gf, sf, explore=True)
    gf, sf, r, done, info = env.step(idx)
    ppo.store_reward(r, done)
    if done:
        gf, sf = env.reset()
    print("Warm-up done.\n")

    # ──────────────────────────────────────────────────────────────────
    # Fine-grained profiling: replicate env.step() with timing
    # ──────────────────────────────────────────────────────────────────
    print(f"--- Profiling {PROFILE_DECISIONS} decisions "
          f"(~{PROFILE_DECISIONS * cfg['selection_interval']} sim steps) ---\n")

    # Reset via env (sets up _obstacle_t, caches, etc.)
    with timer("reset"):
        gf, sf = env.reset()

    for dec in range(PROFILE_DECISIONS):
        # PPO action
        with timer("ppo.get_action"):
            group_idx = ppo.get_action(gf, sf, explore=True)

        group = env.groups[group_idx]

        # ── Replicate env.step() internals with timing ──
        dist_before = list(env._distance)
        reached_before = list(env._reached)
        n_reached_before = sum(reached_before)
        interval_collision = False
        interval_oob = False
        n_new_reached = 0
        env._cached_h = None
        env._cached_attn_rr = None
        env._cached_attn_ro = None

        with torch.no_grad():
            for inner_step in range(cfg["selection_interval"]):
                # 1. prepare_state
                with timer("inner.prepare_state"):
                    robot_state, _ = policy.prepare_state(
                        env._poses, env._distance, env._cos, env._sin,
                        env._collision, env._action, env._goal_positions,
                    )
                    robot_obs = np.array(robot_state)

                # 2. attention + policy_head (combined)
                with timer("inner.attention+policy_head"):
                    raw_actions, h, attn_rr, attn_ro = (
                        env._get_actions_and_embeddings(robot_obs, env._obstacle_states)
                    )
                env._cached_h = h
                env._cached_attn_rr = attn_rr
                env._cached_attn_ro = attn_ro

                # 3. action coupling
                with timer("inner.action_coupling"):
                    action_out = env._actions_for_group(raw_actions, group)

                # 4. sim.step
                with timer("inner.sim_step"):
                    (
                        env._poses, env._distance, env._cos, env._sin,
                        env._collision, goals, env._action, _rewards,
                        _, env._goal_positions, env._obstacle_states,
                    ) = sim.step(action_out, None, None)

                env._step_count += 1

                # 5. check reached + termination
                with timer("inner.check_term"):
                    for i in range(env.num_robots):
                        if not env._reached[i] and env._distance[i] < env.goal_threshold:
                            env._reached[i] = True
                            n_new_reached += 1
                    if any(env._collision):
                        interval_collision = True
                        break
                    from robot_nav.models.MARL.switcher.switcher_env import _outside_of_bounds
                    if _outside_of_bounds(env._poses, sim):
                        interval_oob = True
                        break
                    if all(env._reached):
                        break
                    if env._step_count >= env.max_episode_steps:
                        break

        # 6. reward
        with timer("compute_reward"):
            reward = env._compute_reward(
                group=group,
                dist_before=dist_before,
                dist_after=list(env._distance),
                reached_before=reached_before,
                n_new_reached=n_new_reached,
                n_reached_before=n_reached_before,
                had_collision=interval_collision or interval_oob,
            )

        all_reached = all(env._reached)
        timeout = env._step_count >= env.max_episode_steps
        done = interval_collision or interval_oob or all_reached or timeout

        if done and all_reached:
            frac_time_left = 1.0 - env._step_count / env.max_episode_steps
            reward += 100.0 * (1.0 + frac_time_left)
        if done and timeout and not all_reached:
            reward -= 20.0

        # 7. _build_obs breakdown
        if env._cached_h is not None:
            # Using cache — no embedding recompute
            with timer("build_obs.use_cached_embed"):
                h = env._cached_h
                attn_rr = env._cached_attn_rr
                attn_ro = env._cached_attn_ro
        else:
            with timer("build_obs.get_embeddings_fresh"):
                robot_state, _ = policy.prepare_state(
                    env._poses, env._distance, env._cos, env._sin,
                    env._collision, env._action, env._goal_positions,
                )
                robot_obs = np.array(robot_state)
                h, attn_rr, attn_ro = env._get_embeddings(robot_obs, env._obstacle_states)

        with timer("build_obs.get_extra_features"):
            extra = env._get_extra_features()

        with timer("build_obs.fb_group_features"):
            gf = fb(
                h, groups, h_glob=None,
                attn_rr=attn_rr, attn_ro=attn_ro, extra=extra,
            ).to(device)

        with timer("build_obs.fb_state_features"):
            sf = fb.build_state_features(h, h_glob=None, extra=extra).to(device)

        # Store in PPO buffer
        with timer("ppo.store_reward"):
            ppo.store_reward(reward, done)

        if done:
            with timer("reset"):
                gf, sf = env.reset()

    # ── PPO update ──
    buf_size = len(ppo.buffer.rewards)
    print(f"\n--- PPO update ({cfg['ppo_epochs']} epochs, buf={buf_size}) ---\n")
    with timer("ppo.train"):
        ppo.train(iterations=cfg["ppo_epochs"])

    # ── Report ──
    timer.report("Fine-Grained Switcher Profile")

    # ── Summary ──
    t = timer.totals
    c = timer.counts
    total_sim_steps = c.get("inner.sim_step", 0)
    total_decisions = PROFILE_DECISIONS

    print(f"Total sim steps: {total_sim_steps}")
    print(f"Total decisions: {total_decisions}")
    print()

    # Per-step breakdown
    print("=== PER SIM STEP (avg ms) ===")
    for key in ["inner.prepare_state", "inner.attention+policy_head",
                "inner.action_coupling", "inner.sim_step", "inner.check_term"]:
        n = c.get(key, 1)
        avg = t.get(key, 0) / n * 1000
        print(f"  {key:<35} {avg:.2f} ms/step")

    inner_total = sum(t.get(k, 0) for k in [
        "inner.prepare_state", "inner.attention+policy_head",
        "inner.action_coupling", "inner.sim_step", "inner.check_term",
    ])
    print(f"  {'TOTAL inner loop':<35} {inner_total/total_decisions*1000:.1f} ms/decision")
    print()

    # Per-decision breakdown
    print("=== PER DECISION (avg ms) ===")
    for key in ["build_obs.use_cached_embed", "build_obs.get_embeddings_fresh",
                "build_obs.get_extra_features", "build_obs.fb_group_features",
                "build_obs.fb_state_features", "compute_reward", "ppo.get_action"]:
        n = c.get(key, 0)
        if n > 0:
            avg = t.get(key, 0) / n * 1000
            print(f"  {key:<35} {avg:.2f} ms  (×{n})")
    print()

    # Total per decision
    obs_total = sum(t.get(k, 0) for k in [
        "build_obs.use_cached_embed", "build_obs.get_embeddings_fresh",
        "build_obs.get_extra_features", "build_obs.fb_group_features",
        "build_obs.fb_state_features",
    ])
    dec_total = inner_total + obs_total + t.get("compute_reward", 0) + t.get("ppo.get_action", 0)
    ms_per_dec = dec_total / total_decisions * 1000

    print(f"=== TOTAL: {ms_per_dec:.1f} ms/decision ===")
    print(f"  Inner loop:    {inner_total/dec_total*100:.0f}%  ({inner_total/total_decisions*1000:.1f} ms)")
    print(f"  Build obs:     {obs_total/dec_total*100:.0f}%  ({obs_total/total_decisions*1000:.1f} ms)")
    print(f"  Reward:        {t.get('compute_reward',0)/dec_total*100:.0f}%")
    print(f"  PPO action:    {t.get('ppo.get_action',0)/dec_total*100:.0f}%")
    print()

    # Extrapolate
    rollout_s = ms_per_dec * 500 / 1000
    ppo_s = t.get("ppo.train", 0)
    total_s = rollout_s + ppo_s
    print(f"=== EXTRAPOLATION (500 dec/rollout) ===")
    print(f"  Rollout:  {rollout_s:.1f} s")
    print(f"  PPO:      {ppo_s:.1f} s")
    print(f"  Total:    {total_s:.1f} s/update")
    print(f"  Updates/h: {3600/total_s:.0f}")
    print(f"  5000 updates: {total_s*5000/3600:.1f} h")
    print(f"  2000 updates: {total_s*2000/3600:.1f} h")
    print(f"  1000 updates: {total_s*1000/3600:.1f} h")


if __name__ == "__main__":
    main()
