"""
Simplified test script: compares trained GroupSwitcher vs random group selection.

Modes:
  "switcher" — uses the supervised GroupSwitcher checkpoint (model architecture
               is fully reconstructed from the checkpoint, so no size mismatches).
  "random"   — uniformly random group selection (baseline).

Only size-2 and size-3 groups are used.

Usage:
    python -m robot_nav.scripts.test_switcher_simple
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
from robot_nav.models.MARL.switcher.supervised import (
    GroupFeatureBuilder,
    GroupSwitcher,
)
from robot_nav.models.MARL.switcher.embedding_utils import extract_embeddings
from robot_nav.models.MARL.switcher.config_loader import config_from_dict
from robot_nav.models.MARL.groups.group_generator import generate_all_groups
from robot_nav.models.MARL.groups.action_coupling import actions_for_group
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

from loguru import logger as _loguru_logger
_loguru_logger.disable("irsim")

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # "switcher" or "random"
    "selection_mode": "switcher",

    # Switcher selection strategy: "argmax", "top_k", or "softmax"
    "selection_strategy": "argmax",
    "top_k_selection": 5,
    "softmax_temperature": 0.1,

    # Switcher checkpoint (supervised)
    "switcher_checkpoint": "robot_nav/models/MARL/switcher/runs/switcher/len1500_decouple_min_success/epoch_100.pt",

    # Decentralized policy
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-partial-inactive_epoch210",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive",

    # Evaluation settings
    "test_episodes": 50,
    "max_steps_per_episode": 2000,
    "selection_interval": 10,
    "trials_per_episode": 3,
    "seed": 42,
    "disable_plotting": False,

    # Environment
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "obstacle_proximity_threshold": 1.5,

    # Policy dimensions
    "num_robots": 14,
    "num_obstacles": 7,
    "state_dim": 11,
    "obstacle_state_dim": 4,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# Seeding
# =============================================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Group Generation (size-2 and size-3 only)
# =============================================================================
def generate_groups(num_robots: int) -> List[List[int]]:
    """Generate size-2 and size-3 groups via binary allocation."""
    m = 4 if num_robots <= 14 else 3
    all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)
    return [g for g in all_groups if len(g) in (2, 3)]


# =============================================================================
# Statistics
# =============================================================================
@dataclass
class Stats:
    outcomes: List[str] = field(default_factory=list)        # "success" / "collision" / "timeout"
    steps: List[int] = field(default_factory=list)
    episode_indices: List[int] = field(default_factory=list)

    def record(self, outcome: str, n_steps: int, episode_idx: int):
        self.outcomes.append(outcome)
        self.steps.append(n_steps)
        self.episode_indices.append(episode_idx)

    def summary(self) -> Dict:
        n = len(self.outcomes)
        n_success = self.outcomes.count("success")
        n_collision = self.outcomes.count("collision")
        n_timeout = self.outcomes.count("timeout")

        # Episode-level: success if ANY trial in that episode succeeded
        ep_outcomes: Dict[int, List[str]] = defaultdict(list)
        for ep_idx, outcome in zip(self.episode_indices, self.outcomes):
            ep_outcomes[ep_idx].append(outcome)
        n_ep = len(ep_outcomes)
        n_ep_success = sum(1 for outs in ep_outcomes.values() if "success" in outs)

        success_steps = [s for s, o in zip(self.steps, self.outcomes) if o == "success"]

        return {
            "num_trials": n,
            "num_episodes": n_ep,
            "trial_success_rate": n_success / max(n, 1),
            "trial_collision_rate": n_collision / max(n, 1),
            "trial_timeout_rate": n_timeout / max(n, 1),
            "episode_success_rate": n_ep_success / max(n_ep, 1),
            "avg_steps_success": float(np.mean(success_steps)) if success_steps else 0.0,
            "n_success": n_success,
            "n_collision": n_collision,
            "n_timeout": n_timeout,
            "n_ep_success": n_ep_success,
        }


# =============================================================================
# Helpers
# =============================================================================
def robot_outside_bounds(pose: List[float], sim: MARL_SIM_OBSTACLE) -> bool:
    return (
        pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]
        or pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]
    )


def get_extra_features(
    distance: List[float],
    reached_goal: List[bool],
    poses: List[List[float]],
    goals: List[List[float]],
    sim: MARL_SIM_OBSTACLE,
    current_step: int,
    max_steps: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build the extra feature dict expected by GroupFeatureBuilder."""
    num_robots = sim.num_robots

    dist_to_goal = torch.tensor(distance, dtype=torch.float32, device=device)

    clearances = [sim.get_min_obstacle_clearance(i) for i in range(num_robots)]
    clearance = torch.tensor(clearances, dtype=torch.float32, device=device)

    reached = torch.tensor(
        [1.0 if r else 0.0 for r in reached_goal], dtype=torch.float32, device=device
    )

    heading_errors = []
    for i in range(num_robots):
        dx = goals[i][0] - poses[i][0]
        dy = goals[i][1] - poses[i][1]
        goal_angle = np.arctan2(dy, dx)
        diff = abs(goal_angle - poses[i][2])
        diff = min(diff, 2 * np.pi - diff)
        heading_errors.append(diff)
    heading_error = torch.tensor(heading_errors, dtype=torch.float32, device=device)

    # Global scalars (scalar tensors — GroupFeatureBuilder reads them as tensors)
    var_dist = torch.var(dist_to_goal)
    frac_reached = torch.tensor(
        sum(reached_goal) / max(num_robots, 1), dtype=torch.float32, device=device
    )
    steps_frac = torch.tensor(
        current_step / max(max_steps, 1), dtype=torch.float32, device=device
    )

    return {
        "dist_to_goal": dist_to_goal,
        "clearance": clearance,
        "reached": reached,
        "heading_error": heading_error,
        "var_dist_to_goal": var_dist,
        "frac_reached_global": frac_reached,
        "steps_elapsed_frac": steps_frac,
    }


# =============================================================================
# Switcher Selector
# =============================================================================
class SwitcherSelector:
    """
    Wraps a trained GroupSwitcher for group selection.

    The model architecture (scalar_dim etc.) is read entirely from the
    checkpoint — the caller never needs to specify it manually.
    """

    def __init__(
        self,
        checkpoint_path: str,
        policy: TD3Obstacle,
        groups: List[List[int]],
        device: torch.device,
        selection_strategy: str = "argmax",
        top_k: int = 1,
        softmax_temperature: float = 1.0,
    ):
        self.policy = policy
        self.groups = groups
        self.device = device
        self.selection_strategy = selection_strategy
        self.top_k = max(1, min(top_k, len(groups)))
        self.softmax_temperature = max(0.01, softmax_temperature)

        # ── Load checkpoint ──────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = ckpt.get("config", {})

        # Infer embed_dim from checkpoint weights (embed_tower input = 2*embed_dim)
        embed_tower_w = ckpt["model_state_dict"]["embed_tower.0.weight"]
        embed_dim = embed_tower_w.shape[1] // 2  # e.g. 1024 → 512

        # Infer scalar_dim directly from checkpoint weights
        scalar_tower_w = ckpt["model_state_dict"]["scalar_tower.0.weight"]
        scalar_dim = scalar_tower_w.shape[1]  # e.g. 12

        # Infer hidden dims
        embed_hidden = embed_tower_w.shape[0]               # e.g. 256
        scalar_hidden = scalar_tower_w.shape[0]             # e.g. 64
        fusion_hidden = ckpt["model_state_dict"]["fusion.0.weight"].shape[0]  # e.g. 256
        dropout = model_cfg.get("dropout", 0.1)

        logging.getLogger(__name__).info(
            f"  Checkpoint: embed_dim={embed_dim}, scalar_dim={scalar_dim}, "
            f"embed_hidden={embed_hidden}, scalar_hidden={scalar_hidden}, "
            f"fusion_hidden={fusion_hidden}"
        )

        # ── Reconstruct feature builder from saved switcher_config ────────
        sw_cfg = config_from_dict(ckpt["switcher_config"])
        self.feature_builder = GroupFeatureBuilder.from_config(
            sw_cfg,
            embed_dim=embed_dim,
            pooling=sw_cfg.pooling,
            max_group_size=model_cfg.get("max_group_size", 7),
        )

        # The checkpoint was trained without urgency flag if scalar_dim == base
        # (we don't append urgency — it must match what was used at training time)
        base_scalar_dim = self.feature_builder.scalar_dim
        self.use_urgency_flag = (scalar_dim == base_scalar_dim + 1)
        if self.use_urgency_flag:
            logging.getLogger(__name__).info("  Urgency flag: enabled (detected from checkpoint)")
        else:
            logging.getLogger(__name__).info("  Urgency flag: disabled (detected from checkpoint)")

        # ── Build and load the network ────────────────────────────────────
        self.switcher = GroupSwitcher(
            embed_dim=embed_dim,
            scalar_dim=scalar_dim,
            embed_hidden=embed_hidden,
            scalar_hidden=scalar_hidden,
            fusion_hidden=fusion_hidden,
            dropout=dropout,
        ).to(device)
        self.switcher.load_state_dict(ckpt["model_state_dict"])
        self.switcher.eval()

    @torch.no_grad()
    def select(
        self,
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        extra: Dict[str, torch.Tensor],
    ) -> List[int]:
        h, attn_rr, attn_ro = extract_embeddings(
            self.policy.actor.attention, robot_obs, obstacle_obs, self.device
        )

        X = self.feature_builder(
            h=h, groups=self.groups, h_glob=None,
            attn_rr=attn_rr, attn_ro=attn_ro, extra=extra,
        )  # (M, D_base)

        if self.use_urgency_flag:
            # All urgency flags = 0 (no urgency tracking in this simplified script)
            flags = torch.zeros(len(self.groups), 1, dtype=torch.float32, device=self.device)
            X = torch.cat([X, flags], dim=1)

        logits = self.switcher(X.to(self.device))  # (M,)

        if self.selection_strategy == "argmax":
            idx = logits.argmax().item()
        elif self.selection_strategy == "top_k":
            _, top_idx = torch.topk(logits, k=self.top_k)
            idx = top_idx[random.randrange(self.top_k)].item()
        elif self.selection_strategy == "softmax":
            probs = torch.softmax(logits / self.softmax_temperature, dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()
        else:
            raise ValueError(f"Unknown strategy: {self.selection_strategy}")

        return self.groups[idx]


# =============================================================================
# Evaluation loop
# =============================================================================
def evaluate(
    sim: MARL_SIM_OBSTACLE,
    policy: TD3Obstacle,
    groups: List[List[int]],
    selection_mode: str,
    switcher: Optional[SwitcherSelector],
    num_episodes: int,
    max_steps: int,
    selection_interval: int,
    trials_per_episode: int,
    seed: int,
    device: torch.device,
) -> Stats:
    stats = Stats()
    num_robots = sim.num_robots

    pbar = tqdm(range(num_episodes), desc=f"[{selection_mode}]")
    for episode in pbar:
        for _trial in range(trials_per_episode):
            # Deterministic environment reset per episode
            set_global_seed(seed + episode)
            (
                poses, distance, cos, sin, collision, goals,
                action, reward, positions, goal_positions, obstacle_states,
            ) = sim.reset(random_obstacles=True)

            episode_reward = 0.0
            current_group = None
            reached_goal = [False] * num_robots
            had_collision = False

            for step in range(max_steps):
                if all(reached_goal):
                    break

                # ── Select group ──────────────────────────────────────────
                if step % selection_interval == 0 or current_group is None:
                    robot_state, _ = policy.prepare_state(
                        poses, distance, cos, sin, collision, action, goal_positions
                    )
                    robot_obs = np.array(robot_state)

                    if selection_mode == "switcher" and switcher is not None:
                        extra = get_extra_features(
                            distance, reached_goal, poses, goal_positions,
                            sim, step, max_steps, device,
                        )
                        current_group = switcher.select(robot_obs, obstacle_states, extra)
                    else:
                        current_group = random.choice(groups)

                # ── Actions ───────────────────────────────────────────────
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

                # ── Step ──────────────────────────────────────────────────
                (
                    poses, distance, cos, sin, collision, goals,
                    action, reward, positions, goal_positions, obstacle_states,
                ) = sim.step(action_out, None, None)

                episode_reward += sum(reward)

                # Update reached flags
                for i in range(num_robots):
                    if not reached_goal[i] and goals[i]:
                        reached_goal[i] = True

                # Check collision / out-of-bounds
                if any(collision[i] or robot_outside_bounds(poses[i], sim) for i in range(num_robots)):
                    had_collision = True
                    break

            # ── Outcome ───────────────────────────────────────────────────
            if had_collision:
                outcome = "collision"
            elif all(reached_goal):
                outcome = "success"
            else:
                outcome = "timeout"

            stats.record(outcome, step + 1, episode)

        # Update progress bar
        s = stats.summary()
        pbar.set_postfix({
            "SR": f"{s['trial_success_rate']:.1%}",
            "CR": f"{s['trial_collision_rate']:.1%}",
            "Ep_SR": f"{s['episode_success_rate']:.1%}",
        })

    return stats


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    log = logging.getLogger(__name__)
    cfg = CONFIG

    device = torch.device(cfg["device"])
    log.info(f"Device: {device}")
    log.info(f"Selection mode: {cfg['selection_mode']}")

    # ── Environment ───────────────────────────────────────────────────────
    log.info("Creating simulation environment...")
    sim = MARL_SIM_OBSTACLE(
        world_file=cfg["world_file"],
        disable_plotting=cfg["disable_plotting"],
        reward_phase=5,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=cfg["obstacle_proximity_threshold"],
    )
    log.info(f"  {sim.num_robots} robots, {sim.num_obstacles} obstacles")

    # ── Decentralized policy ──────────────────────────────────────────────
    log.info("Loading decentralized policy...")
    policy = TD3Obstacle(
        state_dim=cfg["state_dim"],
        action_dim=2,
        max_action=1.0,
        device=device,
        num_robots=cfg["num_robots"],
        num_obstacles=cfg["num_obstacles"],
        obstacle_state_dim=cfg["obstacle_state_dim"],
        load_model=True,
        model_name=cfg["decentralized_model_name"],
        load_model_name=cfg["decentralized_model_name"],
        load_directory=Path(cfg["decentralized_model_directory"]),
        save_directory=Path(cfg["decentralized_model_directory"]),
    )

    # ── Groups (size 2 and 3 only) ────────────────────────────────────────
    groups = generate_groups(cfg["num_robots"])
    log.info(f"Candidate groups: {len(groups)} total  "
             f"(size-2: {sum(1 for g in groups if len(g)==2)}, "
             f"size-3: {sum(1 for g in groups if len(g)==3)})")

    # ── Switcher (loaded from checkpoint — architecture auto-detected) ────
    switcher = None
    if cfg["selection_mode"] == "switcher":
        ckpt_path = cfg["switcher_checkpoint"]
        log.info(f"Loading switcher checkpoint: {ckpt_path}")
        switcher = SwitcherSelector(
            checkpoint_path=ckpt_path,
            policy=policy,
            groups=groups,
            device=device,
            selection_strategy=cfg["selection_strategy"],
            top_k=cfg["top_k_selection"],
            softmax_temperature=cfg["softmax_temperature"],
        )
        log.info(f"  Strategy: {cfg['selection_strategy']}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info(
        f"\nEvaluating {cfg['test_episodes']} episodes × "
        f"{cfg['trials_per_episode']} trials = "
        f"{cfg['test_episodes'] * cfg['trials_per_episode']} total trials"
    )
    stats = evaluate(
        sim=sim,
        policy=policy,
        groups=groups,
        selection_mode=cfg["selection_mode"],
        switcher=switcher,
        num_episodes=cfg["test_episodes"],
        max_steps=cfg["max_steps_per_episode"],
        selection_interval=cfg["selection_interval"],
        trials_per_episode=cfg["trials_per_episode"],
        seed=cfg["seed"],
        device=device,
    )

    # ── Results ───────────────────────────────────────────────────────────
    s = stats.summary()
    log.info("\n" + "=" * 60)
    log.info(f"RESULTS  —  mode: {cfg['selection_mode'].upper()}")
    log.info("=" * 60)
    log.info(f"  Total trials:      {s['num_trials']}  ({s['num_episodes']} episodes × {cfg['trials_per_episode']} trials)")
    log.info(f"")
    log.info(f"  Per-trial results:")
    log.info(f"    Success:   {s['n_success']:3d} / {s['num_trials']}  ({s['trial_success_rate']:.1%})")
    log.info(f"    Collision: {s['n_collision']:3d} / {s['num_trials']}  ({s['trial_collision_rate']:.1%})")
    log.info(f"    Timeout:   {s['n_timeout']:3d} / {s['num_trials']}  ({s['trial_timeout_rate']:.1%})")
    log.info(f"")
    log.info(f"  Episode-level (success if ANY trial succeeds):")
    log.info(f"    Success:   {s['n_ep_success']:3d} / {s['num_episodes']}  ({s['episode_success_rate']:.1%})")
    log.info(f"")
    log.info(f"  Avg steps on success: {s['avg_steps_success']:.1f}")
    log.info("=" * 60)

    return stats


if __name__ == "__main__":
    main()
