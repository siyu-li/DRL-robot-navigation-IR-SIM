"""
Test script for the attention-pooling GroupSwitcher (train_switcher_attn.py output).

Two modes:
  "switcher"  — AttnSwitcherGroupSelector with trained GroupSwitcher + AttentionGroupPooling
  "random"    — uniform random group selection (baseline)

Key design decisions that differ from test_switcher.py:
  - Loads BOTH model_state_dict AND attn_pool_state_dict from the checkpoint.
  - Uses AttnGroupFeatureBuilder (no scalars, no urgency, no attn_rr/attn_ro).
  - AttnSwitcherGroupSelector is a clean, minimal class; no extra-feature machinery.
  - GroupSwitcher is always built with scalar_dim=0 (matches training).
  - Episode loop is identical to test_switcher.py v2 semantics:
      * no per-robot goal reset
      * episode ends on collision / all-reach-goal / timeout
      * deterministic per-episode seed for fair comparison

Usage:
    python -m robot_nav.scripts.test_switcher_attn
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.switcher.supervised import GroupSwitcher
from robot_nav.models.MARL.switcher.supervised.attn_feature_builder import AttnGroupFeatureBuilder
from robot_nav.models.MARL.switcher.attention_pooling import AttentionGroupPooling
from robot_nav.models.MARL.switcher.embedding_utils import extract_embeddings
from robot_nav.models.MARL.switcher.config_loader import load_switcher_config, build_attn_pool
from robot_nav.models.MARL.groups.group_generator import generate_all_groups
from robot_nav.models.MARL.groups.action_coupling import actions_for_group
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

# Suppress IRSim warnings
from loguru import logger as _loguru_logger
_loguru_logger.disable("irsim")


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # "switcher" uses the trained attn model; "random" is the baseline
    "selection_mode": "random",

    # Selection strategy for switcher mode
    #   "argmax"  — always pick the highest-scoring group (deterministic)
    #   "softmax" — temperature-scaled softmax sampling
    "selection_strategy": "softmax",
    "softmax_temperature": 0.1,

    # Reproducibility
    "seed": 42,
    "trials_per_episode": 3,

    # Switcher checkpoint (saved by train_switcher_attn.py)
    "switcher_checkpoint": "robot_nav/models/MARL/switcher/runs/switcher_attn/epoch_20.pt",

    # Switcher YAML — must have pooling: "attention"
    "switcher_config_path": "robot_nav/models/MARL/switcher/switcher_config.yaml",

    # Decentralized TD3 policy
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-partial-inactive_epoch210",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive",

    # Evaluation
    "test_episodes": 50,
    "max_steps_per_episode": 1000,
    "selection_interval": 10,
    "disable_plotting": False,

    # Group generation (must match training data)
    "include_size_1": False,
    "include_size_2": True,
    "include_size_3": True,
    "include_size_4": False,
    "include_size_7": False,

    # Policy / env dimensions
    "num_robots": 14,
    "num_obstacles": 7,
    "state_dim": 11,
    "obstacle_state_dim": 4,
    "embedding_dim": 256,   # GAT output per robot; embed_dim = 2 * 256 = 512

    # World
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "obstacle_proximity_threshold": 1.5,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# Seeding
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Group generation
# =============================================================================

def generate_candidate_groups(
    num_robots: int,
    include_size_1: bool = False,
    include_size_2: bool = True,
    include_size_3: bool = True,
    include_size_4: bool = False,
    include_size_7: bool = False,
) -> List[List[int]]:
    if num_robots <= 6:
        m = 3
    elif num_robots <= 14:
        m = 4
    else:
        raise ValueError(f"Unsupported num_robots={num_robots}.")

    all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)

    allowed = set()
    if include_size_1: allowed.add(1)
    if include_size_2: allowed.add(2)
    if include_size_3: allowed.add(3)
    if include_size_4: allowed.add(4)
    if include_size_7: allowed.add(7)

    return [g for g in all_groups if len(g) in allowed]


# =============================================================================
# Bounds check
# =============================================================================

def robot_outside_bounds(pose: List[float], sim: MARL_SIM_OBSTACLE) -> bool:
    if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
        return True
    if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
        return True
    return False


# =============================================================================
# Statistics
# =============================================================================

@dataclass
class TestStatistics:
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    episode_outcomes: List[str] = field(default_factory=list)
    trial_episode_indices: List[int] = field(default_factory=list)
    success_episode_path_lengths: List[List[float]] = field(default_factory=list)

    executed_group_counts: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    executed_size_counts: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_collision_counts: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_execution_counts: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_intra_collisions: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_extra_robot_collisions: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    group_obstacle_collisions: Dict[Tuple[int, ...], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_group_execution(self, group: List[int]) -> None:
        key = tuple(group)
        self.executed_group_counts[key] += 1
        self.executed_size_counts[len(group)] += 1
        self.group_execution_counts[key] += 1

    def record_collision(
        self,
        group: List[int],
        collision_indices: List[int],
        collision_types: Dict[int, str],
    ) -> None:
        key = tuple(group)
        group_set = set(group)
        for robot_idx in collision_indices:
            if robot_idx in group_set:
                self.group_collision_counts[key] += 1
                ctype = collision_types.get(robot_idx, "obstacle")
                if ctype == "intra":
                    self.group_intra_collisions[key] += 1
                elif ctype == "extra_robot":
                    self.group_extra_robot_collisions[key] += 1
                else:
                    self.group_obstacle_collisions[key] += 1

    def record_episode(
        self,
        total_reward: float,
        steps: int,
        outcome: str,
        episode_idx: int,
        path_lengths: Optional[List[float]] = None,
    ) -> None:
        self.episode_rewards.append(float(total_reward))
        self.episode_steps.append(int(steps))
        self.episode_outcomes.append(outcome)
        self.trial_episode_indices.append(episode_idx)
        if outcome == "success" and path_lengths is not None:
            self.success_episode_path_lengths.append(list(path_lengths))

    def get_summary(self, trials_per_episode: int = 1) -> Dict:
        num_trials = len(self.episode_rewards)
        num_episodes = (
            len(set(self.trial_episode_indices)) if self.trial_episode_indices else num_trials
        )
        total_exec = sum(self.executed_size_counts.values())

        trial_success  = sum(1 for o in self.episode_outcomes if o == "success")
        trial_collision = sum(1 for o in self.episode_outcomes if o == "collision")
        trial_timeout  = sum(1 for o in self.episode_outcomes if o == "timeout")

        # Episode-level: success if ANY trial of that episode succeeded
        ep_outcomes: Dict[int, List[str]] = {}
        for ep_idx, outcome in zip(self.trial_episode_indices, self.episode_outcomes):
            ep_outcomes.setdefault(ep_idx, []).append(outcome)
        episode_success_count = sum(1 for v in ep_outcomes.values() if "success" in v)

        success_rewards = [r for r, o in zip(self.episode_rewards, self.episode_outcomes) if o == "success"]
        collision_rewards = [r for r, o in zip(self.episode_rewards, self.episode_outcomes) if o == "collision"]
        timeout_rewards  = [r for r, o in zip(self.episode_rewards, self.episode_outcomes) if o == "timeout"]
        success_steps    = [s for s, o in zip(self.episode_steps,   self.episode_outcomes) if o == "success"]

        # Path length stats
        if self.success_episode_path_lengths:
            path_arr = np.array(self.success_episode_path_lengths)
            per_robot_avg = path_arr.mean(axis=0).tolist()
            per_robot_std = path_arr.std(axis=0).tolist()
            total_avg = float(path_arr.mean())
            total_std = float(path_arr.std())
        else:
            per_robot_avg = per_robot_std = []
            total_avg = total_std = 0.0

        # Collision rate per size
        collision_rate_by_size = {}
        for size in [1, 2, 3, 4, 7]:
            gs = [g for g in self.group_execution_counts if len(g) == size]
            tex = sum(self.group_execution_counts[g] for g in gs)
            tco = sum(self.group_collision_counts.get(g, 0) for g in gs)
            collision_rate_by_size[size] = tco / max(tex, 1)

        collision_rates = {
            g: self.group_collision_counts.get(g, 0) / self.group_execution_counts[g]
            for g in self.group_execution_counts
        }

        return {
            "num_episodes":             num_episodes,
            "num_trials":               num_trials,
            "trials_per_episode":       trials_per_episode,
            "total_executions":         total_exec,
            "avg_episode_reward":       float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "avg_steps":                float(np.mean(self.episode_steps))   if self.episode_steps   else 0.0,
            "trial_success_count":      trial_success,
            "trial_collision_count":    trial_collision,
            "trial_timeout_count":      trial_timeout,
            "trial_success_rate":       trial_success  / max(num_trials, 1),
            "trial_collision_rate":     trial_collision / max(num_trials, 1),
            "trial_timeout_rate":       trial_timeout  / max(num_trials, 1),
            "episode_success_count":    episode_success_count,
            "episode_success_rate":     episode_success_count / max(num_episodes, 1),
            "avg_success_reward":       float(np.mean(success_rewards))  if success_rewards  else 0.0,
            "avg_collision_reward":     float(np.mean(collision_rewards)) if collision_rewards else 0.0,
            "avg_timeout_reward":       float(np.mean(timeout_rewards))  if timeout_rewards  else 0.0,
            "avg_success_steps":        float(np.mean(success_steps))    if success_steps    else 0.0,
            "per_robot_avg_path_length": per_robot_avg,
            "per_robot_std_path_length": per_robot_std,
            "total_avg_path_length":    total_avg,
            "total_std_path_length":    total_std,
            "num_success_episodes_for_path": len(self.success_episode_path_lengths),
            "size_distribution":        {s: c / max(total_exec, 1) for s, c in self.executed_size_counts.items()},
            "top_10_executed_groups":   sorted(self.executed_group_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "collision_rate_by_size":   collision_rate_by_size,
            "top_10_collision_groups":  sorted(collision_rates.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_10_safest_groups":     sorted(collision_rates.items(), key=lambda x: x[1])[:10],
        }


# =============================================================================
# Collision classifier
# =============================================================================

def classify_collisions(
    collision: List[bool],
    poses: List[List[float]],
    group: List[int],
    sim: MARL_SIM_OBSTACLE,
    robot_collision_threshold: float = 0.6,
) -> Dict[int, str]:
    collision_types: Dict[int, str] = {}
    group_set = set(group)
    num_robots = len(collision)

    for i, is_collided in enumerate(collision):
        if not is_collided:
            continue

        closest_dist = float("inf")
        closest_idx = -1
        for j in range(num_robots):
            if i == j:
                continue
            d = np.sqrt((poses[i][0] - poses[j][0]) ** 2 + (poses[i][1] - poses[j][1]) ** 2)
            if d < closest_dist:
                closest_dist = d
                closest_idx = j

        obstacle_clearance = sim.get_min_obstacle_clearance(i)

        if closest_dist < robot_collision_threshold:
            if i in group_set and closest_idx in group_set:
                collision_types[i] = "intra"
            else:
                collision_types[i] = "extra_robot"
        elif obstacle_clearance < 0.4:
            collision_types[i] = "obstacle"
        else:
            collision_types[i] = "obstacle"

    return collision_types


# =============================================================================
# Attention Switcher Group Selector
# =============================================================================

class AttnSwitcherGroupSelector:
    """
    Selects groups using a trained GroupSwitcher + AttentionGroupPooling.

    Differences from the old SwitcherGroupSelector:
      - Holds attn_pool as a separate module; both are loaded from the checkpoint.
      - Uses AttnGroupFeatureBuilder: only needs (h, groups) — no scalars, no
        urgency, no attn_rr/attn_ro.
      - GroupSwitcher is built with scalar_dim=0 (matches training).
    """

    def __init__(
        self,
        model: GroupSwitcher,
        attn_pool: AttentionGroupPooling,
        policy: TD3Obstacle,
        groups: List[List[int]],
        device: torch.device,
        selection_strategy: str = "argmax",
        softmax_temperature: float = 0.1,
    ):
        self.model = model.to(device)
        self.model.eval()

        self.attn_pool = attn_pool.to(device)
        self.attn_pool.eval()

        # feature builder wraps attn_pool — no scalars
        self.feature_builder = AttnGroupFeatureBuilder(
            attn_pool=self.attn_pool,
            embed_dim=attn_pool.embed_dim,
        )

        self.policy = policy
        self.groups = groups
        self.device = device

        if selection_strategy not in ("argmax", "softmax"):
            raise ValueError(
                f"selection_strategy must be 'argmax' or 'softmax', got '{selection_strategy}'"
            )
        self.selection_strategy = selection_strategy
        self.softmax_temperature = max(1e-2, softmax_temperature)

    @torch.no_grad()
    def select_group(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
    ) -> List[int]:
        """
        Select a group given current observations.

        Args:
            robot_obs:    (N, state_dim) robot observations.
            obstacle_obs: (N_obs, obs_dim) obstacle observations.

        Returns:
            Selected group as List[int].
        """
        # Get per-robot embeddings from the frozen policy's attention module.
        # attn_rr and attn_ro are returned for API compatibility but not used.
        h, _attn_rr, _attn_ro = extract_embeddings(
            self.policy.actor.attention, robot_obs, obstacle_obs, self.device
        )

        # Build (M, 2*embed_dim) feature matrix — no scalars, no extra dict
        X = self.feature_builder(h=h, groups=self.groups)  # (M, 2*embed_dim)
        X = X.to(self.device)

        # Score all groups
        logits = self.model(X)  # (M,)

        if self.selection_strategy == "argmax":
            selected_idx = int(logits.argmax().item())
        else:  # softmax
            scaled = logits / self.softmax_temperature
            probs = torch.softmax(scaled, dim=0)
            selected_idx = int(torch.multinomial(probs, num_samples=1).item())

        return self.groups[selected_idx]


# =============================================================================
# Evaluation loop
# =============================================================================

def run_test_evaluation(
    sim: MARL_SIM_OBSTACLE,
    policy: TD3Obstacle,
    groups: List[List[int]],
    selection_mode: str,
    switcher_selector: Optional[AttnSwitcherGroupSelector],
    num_episodes: int = 50,
    max_steps: int = 5000,
    selection_interval: int = 10,
    trials_per_episode: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> TestStatistics:
    """
    Episode semantics (identical to test_switcher.py v2):
      - Full env reset at the start of every trial.
      - Same episode seed → same obstacle/robot config across trials.
      - Episode ends: success (all reach goal) / collision / timeout.
    """
    stats = TestStatistics()
    num_robots = sim.num_robots

    pbar = (
        tqdm(range(num_episodes), desc=f"Testing ({selection_mode})")
        if verbose else range(num_episodes)
    )

    for episode in pbar:
        for trial in range(trials_per_episode):
            # Deterministic env config per episode (same for all trials)
            set_global_seed(seed + episode)

            (
                poses, distance, cos, sin, collision, goals,
                action, reward, positions, goal_positions, obstacle_states,
            ) = sim.reset(random_obstacles=True)

            episode_reward = 0.0
            current_group = None
            steps = 0
            episode_had_collision = False
            reached_goal = [False] * num_robots
            path_lengths = [0.0] * num_robots
            prev_positions = [[poses[i][0], poses[i][1]] for i in range(num_robots)]

            while steps < max_steps:
                # ── Check success ──────────────────────────────────────
                if all(reached_goal):
                    break

                # ── Select group ───────────────────────────────────────
                if steps % selection_interval == 0 or current_group is None:
                    robot_state, _ = policy.prepare_state(
                        poses, distance, cos, sin, collision, action, goal_positions
                    )
                    robot_obs = np.array(robot_state)

                    if selection_mode == "switcher" and switcher_selector is not None:
                        current_group = switcher_selector.select_group(
                            robot_obs=robot_obs,
                            obstacle_obs=obstacle_states,
                        )
                    else:
                        current_group = random.choice(groups)

                    stats.record_group_execution(current_group)

                # ── Get actions ────────────────────────────────────────
                robot_state, _ = policy.prepare_state(
                    poses, distance, cos, sin, collision, action, goal_positions
                )
                robot_obs = np.array(robot_state)

                action_out = actions_for_group(
                    policy=policy,
                    robot_obs=robot_obs,
                    obstacle_obs=obstacle_states,
                    group=current_group,
                    num_robots=num_robots,
                )

                # ── Step ───────────────────────────────────────────────
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states,
                ) = sim.step(action_out, None, None)

                steps += 1
                episode_reward += sum(reward)

                # ── Update path lengths ────────────────────────────────
                for i in range(num_robots):
                    dx = poses[i][0] - prev_positions[i][0]
                    dy = poses[i][1] - prev_positions[i][1]
                    path_lengths[i] += np.sqrt(dx * dx + dy * dy)
                    prev_positions[i] = [poses[i][0], poses[i][1]]

                # ── Latch reached goals ────────────────────────────────
                for i in range(num_robots):
                    if not reached_goal[i] and goals[i]:
                        reached_goal[i] = True

                # ── Collision / OOB check ──────────────────────────────
                collided = [
                    i for i in range(num_robots)
                    if collision[i] or robot_outside_bounds(poses[i], sim)
                ]
                if collided:
                    episode_had_collision = True
                    ctypes = classify_collisions(collision, poses, current_group, sim)
                    stats.record_collision(current_group, collided, ctypes)
                    break

            # ── Record trial outcome ───────────────────────────────────
            if episode_had_collision:
                outcome = "collision"
            elif all(reached_goal):
                outcome = "success"
            else:
                outcome = "timeout"

            stats.record_episode(
                total_reward=episode_reward,
                steps=steps,
                outcome=outcome,
                episode_idx=episode,
                path_lengths=path_lengths,
            )

        if verbose and isinstance(pbar, tqdm):
            n_trials = len(stats.episode_outcomes)
            n_succ = sum(1 for o in stats.episode_outcomes if o == "success")
            ep_outcomes: Dict[int, List[str]] = {}
            for ep_idx, o in zip(stats.trial_episode_indices, stats.episode_outcomes):
                ep_outcomes.setdefault(ep_idx, []).append(o)
            ep_succ = sum(1 for v in ep_outcomes.values() if "success" in v)
            pbar.set_postfix({
                "T_S": f"{n_succ}/{n_trials}",
                "E_S": f"{ep_succ}/{episode + 1}",
            })

    return stats


# =============================================================================
# Pretty-print summary
# =============================================================================

def print_summary(stats: TestStatistics, selection_mode: str, trials_per_episode: int) -> None:
    log = logging.getLogger(__name__)
    summary = stats.get_summary(trials_per_episode=trials_per_episode)

    log.info("\n" + "=" * 70)
    log.info(f"ATTN SWITCHER TEST RESULTS — mode: {selection_mode.upper()}")
    log.info("=" * 70)
    log.info(f"  Episodes: {summary['num_episodes']}  |  "
             f"Trials/ep: {summary['trials_per_episode']}  |  "
             f"Total trials: {summary['num_trials']}")
    log.info(f"  Avg reward/trial: {summary['avg_episode_reward']:.2f}  |  "
             f"Avg steps/trial: {summary['avg_steps']:.1f}")

    log.info("\n--- Trial-Level Results ---")
    log.info(f"  Success:   {summary['trial_success_count']}/{summary['num_trials']} "
             f"({summary['trial_success_rate']:.2%})")
    log.info(f"  Collision: {summary['trial_collision_count']}/{summary['num_trials']} "
             f"({summary['trial_collision_rate']:.2%})")
    log.info(f"  Timeout:   {summary['trial_timeout_count']}/{summary['num_trials']} "
             f"({summary['trial_timeout_rate']:.2%})")

    log.info("\n--- Episode-Level Results (success if ANY trial succeeds) ---")
    log.info(f"  Success episodes: {summary['episode_success_count']}/{summary['num_episodes']} "
             f"({summary['episode_success_rate']:.2%})")

    log.info("\n--- Avg reward / steps by outcome ---")
    log.info(f"  Success:   reward={summary['avg_success_reward']:.2f}  steps={summary['avg_success_steps']:.1f}")
    log.info(f"  Collision: reward={summary['avg_collision_reward']:.2f}")
    log.info(f"  Timeout:   reward={summary['avg_timeout_reward']:.2f}")

    log.info("\n--- Path length (successful trials only) ---")
    if summary["num_success_episodes_for_path"] > 0:
        log.info(f"  Based on {summary['num_success_episodes_for_path']} successful trials")
        for i, (avg_pl, std_pl) in enumerate(
            zip(summary["per_robot_avg_path_length"], summary["per_robot_std_path_length"])
        ):
            log.info(f"    Robot {i:2d}: {avg_pl:.3f} ± {std_pl:.3f}")
        log.info(f"  Grand avg: {summary['total_avg_path_length']:.3f} "
                 f"± {summary['total_std_path_length']:.3f}")
    else:
        log.info("  No successful trials.")

    log.info("\n--- Group size distribution ---")
    for size, pct in sorted(summary["size_distribution"].items()):
        log.info(f"  Size-{size}: {pct:.1%}")

    log.info("\n--- Collision rate by group size ---")
    for size, rate in sorted(summary["collision_rate_by_size"].items()):
        log.info(f"  Size-{size}: {rate:.2%}")

    log.info("\n--- Top 10 executed groups ---")
    for group, count in summary["top_10_executed_groups"]:
        pct = count / max(summary["total_executions"], 1) * 100
        log.info(f"  {list(group)} (size-{len(group)}): {count}×  ({pct:.1f}%)")

    log.info("\n--- Top 10 safest groups ---")
    for group, rate in summary["top_10_safest_groups"]:
        log.info(f"  {list(group)} (size-{len(group)}): {rate:.2%}")

    log.info("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    config = CONFIG
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    log = logging.getLogger(__name__)

    device = torch.device(config["device"])
    log.info(f"Device: {device}  |  Mode: {config['selection_mode']}")

    # ── Simulation environment ─────────────────────────────────────────────
    log.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
        reward_phase=5,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )
    log.info(f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles")

    # ── Decentralized policy ───────────────────────────────────────────────
    log.info("Loading decentralized policy...")
    policy = TD3Obstacle(
        state_dim=config["state_dim"],
        action_dim=2,
        max_action=1.0,
        device=device,
        num_robots=config["num_robots"],
        num_obstacles=config["num_obstacles"],
        obstacle_state_dim=config["obstacle_state_dim"],
        load_model=True,
        model_name=config["decentralized_model_name"],
        load_model_name=config["decentralized_model_name"],
        load_directory=Path(config["decentralized_model_directory"]),
        save_directory=Path(config["decentralized_model_directory"]),
    )

    # ── Candidate groups ───────────────────────────────────────────────────
    groups = generate_candidate_groups(
        num_robots=config["num_robots"],
        include_size_1=config.get("include_size_1", False),
        include_size_2=config.get("include_size_2", True),
        include_size_3=config.get("include_size_3", True),
        include_size_4=config.get("include_size_4", False),
        include_size_7=config.get("include_size_7", False),
    )
    log.info(f"Candidate groups: {len(groups)}  "
             f"(size-2: {sum(1 for g in groups if len(g)==2)}, "
             f"size-3: {sum(1 for g in groups if len(g)==3)})")

    # ── Load attn switcher if needed ───────────────────────────────────────
    switcher_selector: Optional[AttnSwitcherGroupSelector] = None

    if config["selection_mode"] == "switcher":
        ckpt_path = Path(config["switcher_checkpoint"])
        if not ckpt_path.exists():
            log.error(f"Checkpoint not found: {ckpt_path}")
            log.error("Train first with train_switcher_attn.py or switch to 'random' mode.")
            return

        log.info(f"Loading attn switcher from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Recover embed_dim from checkpoint config; fall back to 2 × embedding_dim
        model_cfg = ckpt.get("config", {})
        embed_dim = model_cfg.get("embed_dim", config["embedding_dim"] * 2)

        # Rebuild attn_pool from the YAML (architecture parameters)
        sw_cfg = load_switcher_config(config["switcher_config_path"])
        if sw_cfg.pooling != "attention":
            raise ValueError(
                f"switcher_config.yaml has pooling='{sw_cfg.pooling}'; "
                "this script requires pooling='attention'."
            )
        attn_pool = build_attn_pool(sw_cfg, embed_dim=embed_dim)

        # Rebuild GroupSwitcher with scalar_dim=0 (no scalars in attn training)
        model = GroupSwitcher(
            embed_dim=embed_dim,
            scalar_dim=0,                                     # ← always 0 for attn model
            embed_hidden=model_cfg.get("embed_hidden", 256),
            fusion_hidden=model_cfg.get("fusion_hidden", 256),
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Load BOTH state dicts
        model.load_state_dict(ckpt["model_state_dict"])
        attn_pool.load_state_dict(ckpt["attn_pool_state_dict"])
        log.info(f"  embed_dim={embed_dim}, scalar_dim=0")
        log.info(f"  attn heads={sw_cfg.attn_n_heads}, score_hidden={sw_cfg.attn_score_hidden}")

        switcher_selector = AttnSwitcherGroupSelector(
            model=model,
            attn_pool=attn_pool,
            policy=policy,
            groups=groups,
            device=device,
            selection_strategy=config.get("selection_strategy", "argmax"),
            softmax_temperature=config.get("softmax_temperature", 0.1),
        )
        log.info(f"  strategy={config.get('selection_strategy')}  "
                 f"temperature={config.get('softmax_temperature')}")

    # ── Run evaluation ─────────────────────────────────────────────────────
    log.info(f"\nRunning {config['test_episodes']} episodes "
             f"× {config.get('trials_per_episode', 1)} trials ...")

    stats = run_test_evaluation(
        sim=sim,
        policy=policy,
        groups=groups,
        selection_mode=config["selection_mode"],
        switcher_selector=switcher_selector,
        num_episodes=config["test_episodes"],
        max_steps=config["max_steps_per_episode"],
        selection_interval=config["selection_interval"],
        trials_per_episode=config.get("trials_per_episode", 1),
        seed=config["seed"],
        verbose=True,
    )

    print_summary(stats, config["selection_mode"], config.get("trials_per_episode", 1))


if __name__ == "__main__":
    main()
