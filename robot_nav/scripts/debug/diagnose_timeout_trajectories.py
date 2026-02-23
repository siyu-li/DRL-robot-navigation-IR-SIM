"""
Diagnostic Trajectory Recording for Timeout Investigation.

Runs the same oracle-based group switching as
collect_oracle_data_singleprocess_allrobotreach.py, but instead of
collecting training data, records full trajectory diagnostics:

  - Per-step: all oracle group scores, selected group, per-robot raw
    policy actions, applied (coupled) actions, poses/distances before
    and after, reached flags, urgency flags, collisions.
  - Per-episode: initial robot/obstacle configuration, outcome
    (all_reached / timeout / collision), per-robot reach step,
    total steps, which robots never reached.

Saved as a .pt file that can be loaded for offline replay and analysis.

Usage:
    python -m robot_nav.scripts.debug.diagnose_timeout_trajectories
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger as loguru_logger
loguru_logger.disable("irsim")

from robot_nav.models.MARL.groups.group_generator import (
    generate_all_groups,
    filter_groups_by_size,
)
from robot_nav.models.MARL.groups.action_coupling import (
    actions_for_group as _shared_actions_for_group,
)

# =============================================================================
# Configuration — mirrors collect_oracle_data_singleprocess_allrobotreach
# =============================================================================
CONFIG = {
    # Output
    "output_path": "robot_nav/models/MARL/switcher/data/diagnostic_trajectories.pt",

    # Episode settings
    "n_samples": 3000,              # Total Phase-1/Phase-2 sample steps to run
    "n_robots": 14,
    "n_obstacles": 7,
    "seed": 42,

    # Oracle settings (same as allrobotreach)
    "oracle_horizon": 10,
    "n_rollouts_per_group": 1,

    # Group generation (same as allrobotreach)
    "include_size_1": True,
    "include_size_2": True,
    "include_size_3": True,
    "include_size_4": True,
    "include_size_7": True,
    "use_rotation_coupling": True,

    # Model
    "state_dim": 11,
    "obstacle_state_dim": 4,
    "decentralized_model_name": "TD3-MARL-obstacle-14robots-gpu_epoch800",
    "decentralized_model_directory": "robot_nav/models/MARL/marlTD3/checkpoint/Feb.10_obstacle_14robot_transfer_gpu",

    # Simulation
    "world_file": "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
    "disable_plotting": True,
    "obstacle_proximity_threshold": 1.5,
    "max_steps_per_episode": 1500,
    "goal_reach_threshold": 0.3,

    # Scoring weights (kept for oracle evaluation — same as allrobotreach)
    "k_reach": 50.0,
    "k_progress": 3.0,
    "k_rotation_progress": 2.0,
    "k_sync": 3.0,
    "k_urgency": 15.0,
    "min_displacement_threshold": 0.2,
    "min_rotation_threshold": 0.1,

    # Urgency tracking
    "urgency_lookback_window": 20,
    "urgency_stuck_threshold": 0.3,

    # Phase 2 selection (same as allrobotreach)
    "phase2_selection": "softmax",
    "phase2_temperature": 0.1,
}


# =============================================================================
# Reuse helpers from collect_oracle_data_singleprocess_allrobotreach
# =============================================================================
def generate_candidate_groups(
    num_robots: int,
    include_size_1: bool = True,
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
        raise ValueError(f"Unsupported number of robots: {num_robots}")
    all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)
    allowed_sizes = set()
    if include_size_1: allowed_sizes.add(1)
    if include_size_2: allowed_sizes.add(2)
    if include_size_3: allowed_sizes.add(3)
    if include_size_4: allowed_sizes.add(4)
    if include_size_7: allowed_sizes.add(7)
    return [g for g in all_groups if len(g) in allowed_sizes]


def outside_of_bounds(poses, sim) -> bool:
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


# =============================================================================
# Snapshot (same as allrobotreach)
# =============================================================================
@dataclass
class SimulationSnapshot:
    robot_states: List[np.ndarray]
    robot_goals: List[np.ndarray]
    prev_distances: List[Optional[float]]

    @classmethod
    def from_sim(cls, sim):
        robot_states = []
        robot_goals = []
        for robot in sim.env.robot_list:
            robot_states.append(robot.state.copy())
            robot_goals.append(robot.goal.copy())
        return cls(
            robot_states=robot_states,
            robot_goals=robot_goals,
            prev_distances=sim.prev_distances.copy(),
        )

    def restore_to_sim(self, sim):
        for i, robot in enumerate(sim.env.robot_list):
            robot.set_state(state=self.robot_states[i], init=True)
            robot.set_goal(self.robot_goals[i], init=True)
        sim.prev_distances = self.prev_distances.copy()


# =============================================================================
# Lightweight oracle evaluator (replicates scoring from allrobotreach exactly)
# =============================================================================
class OracleEvaluator:
    """
    Mirrors OracleDataCollector from allrobotreach, but only exposes the
    pieces needed for trajectory diagnosis: get_action_for_group and
    evaluate_all_groups (which returns per-group scores).
    """

    def __init__(self, sim, policy, groups, horizon, n_rollouts_per_group, device):
        self.sim = sim
        self.policy = policy
        self.groups = groups
        self.horizon = horizon
        self.n_rollouts_per_group = n_rollouts_per_group
        self.device = device
        self.num_robots = sim.num_robots
        self.num_obstacles = sim.num_obstacles

    # ----- action coupling (delegates to shared module) -----
    def get_action_for_group(self, robot_obs, obstacle_obs, group):
        action, _ = self.policy.get_action(robot_obs, obstacle_obs, add_noise=False)
        from robot_nav.models.MARL.groups.action_coupling import actions_for_group_from_raw
        a_out = actions_for_group_from_raw(
            raw_actions=action,
            group=group,
            num_robots=self.num_robots,
            use_rotation_coupling=CONFIG.get("use_rotation_coupling", True),
            rotation_coupling_threshold=3,
        )
        return a_out, action  # also return raw policy action

    # ----- raw policy action for all robots (no coupling, no scaling) -----
    def get_raw_policy_action(self, robot_obs, obstacle_obs):
        """Return raw [-1,1] actions for all robots."""
        action, _ = self.policy.get_action(robot_obs, obstacle_obs, add_noise=False)
        return action  # (N, 2)

    # ----- single-group oracle rollout (identical logic to allrobotreach) -----
    def _evaluate_group_once(self, group, poses, distance, cos_, sin_,
                             collision, action, goal_positions,
                             obstacle_states, snapshot, reached,
                             urgency_flags):
        N = self.num_robots
        goal_threshold = CONFIG.get("goal_reach_threshold", 0.3)
        if reached is None:
            reached = [False] * N
        reached_before = list(reached)
        n_already_reached = sum(reached_before)
        had_collision = False
        n_new_reached = 0
        rollout_reached = list(reached_before)

        initial_poses = [p.copy() for p in poses]
        initial_distances = list(distance)
        initial_obstacle_states = obstacle_states.copy()

        curr_poses = [p.copy() for p in poses]
        curr_distance = list(distance)
        curr_cos = list(cos_)
        curr_sin = list(sin_)
        curr_collision = list(collision)
        curr_action = [a.copy() for a in action]
        curr_goal_positions = [g.copy() for g in goal_positions]
        curr_obstacle_states = obstacle_states.copy()

        for step in range(self.horizon):
            robot_state, _ = self.policy.prepare_state(
                curr_poses, curr_distance, curr_cos, curr_sin,
                curr_collision, curr_action, curr_goal_positions,
            )
            a_in, _ = self.get_action_for_group(
                np.array(robot_state), curr_obstacle_states, group,
            )
            (curr_poses, curr_distance, curr_cos, curr_sin,
             curr_collision, curr_goal, curr_action, reward,
             _, curr_goal_positions, curr_obstacle_states
             ) = self.sim.step(a_in, None, None)

            for i in range(N):
                if not rollout_reached[i] and curr_distance[i] < goal_threshold:
                    rollout_reached[i] = True
                    n_new_reached += 1
            if any(curr_collision[i] for i in group):
                had_collision = True
                break
            if outside_of_bounds(curr_poses, self.sim):
                had_collision = True
                break

        final_poses = [p.copy() for p in curr_poses]
        final_distances = list(curr_distance)
        final_obstacle_states = curr_obstacle_states.copy()

        score = self._compute_trajectory_score(
            group, initial_poses, final_poses,
            initial_distances, final_distances,
            initial_obstacle_states, final_obstacle_states,
            had_collision, n_new_reached, n_already_reached,
            reached_before, goal_positions, urgency_flags,
        )

        snapshot.restore_to_sim(self.sim)
        return score

    # ----- trajectory score (replica of allrobotreach compute_trajectory_score) -----
    def _compute_trajectory_score(
        self, group, initial_poses, final_poses,
        initial_distances, final_distances,
        initial_obstacle_states, final_obstacle_states,
        had_collision, n_new_reached, n_already_reached,
        reached_before, goal_positions, urgency_flags,
    ):
        N = self.num_robots
        k_reach = CONFIG["k_reach"]
        k_progress = CONFIG["k_progress"]
        k_sync = CONFIG["k_sync"]
        k_urgency = CONFIG["k_urgency"]
        k_rotation_progress = CONFIG.get("k_rotation_progress", 2.0)

        if had_collision:
            return -50.0

        # coupled group bonus
        any_in_group_reached = any(reached_before[i] for i in group)
        score = 5.0 if (len(group) > 3 and not any_in_group_reached) else 0.0

        # reach bonus
        if n_new_reached > 0:
            n_rem = N - n_already_reached
            if n_rem > 0:
                for r in range(n_new_reached):
                    rem = n_rem - r
                    if rem > 0:
                        score += k_reach / rem

        # progress (laggard-weighted)
        unreached = [i for i in range(N) if not reached_before[i]]
        if unreached:
            u_dists = [initial_distances[i] for i in unreached]
            mean_d = max(np.mean(u_dists), 0.1)
            for i in group:
                if reached_before[i]:
                    continue
                prog = initial_distances[i] - final_distances[i]
                w = initial_distances[i] / mean_d
                score += k_progress * prog * w

        # rotation progress
        for i in group:
            if reached_before[i]:
                continue
            xi0, yi0, th0 = initial_poses[i]
            xi1, yi1, th1 = final_poses[i]
            gx, gy = goal_positions[i]
            ang0 = np.arctan2(gy - yi0, gx - xi0)
            he0 = min(abs(th0 - ang0), 2 * np.pi - abs(th0 - ang0))
            ang1 = np.arctan2(gy - yi1, gx - xi1)
            he1 = min(abs(th1 - ang1), 2 * np.pi - abs(th1 - ang1))
            score += k_rotation_progress * (he0 - he1)

        # urgency bonus (single-robot only)
        if urgency_flags is not None and len(group) == 1:
            ri = group[0]
            if not reached_before[ri] and urgency_flags[ri]:
                xi0, yi0, _ = initial_poses[ri]
                xi1, yi1, _ = final_poses[ri]
                disp = np.sqrt((xi1 - xi0)**2 + (yi1 - yi0)**2)
                prog = initial_distances[ri] - final_distances[ri]
                if disp > 0.01:
                    ub = k_urgency * disp
                    if prog > 0:
                        ub += k_urgency * prog * 0.5
                    score += ub

        # sync reward (only when ≥1 reached)
        if N >= 2 and any(reached_before):
            var_b = np.var(initial_distances)
            var_a = np.var(final_distances)
            score += k_sync * (var_b - var_a)

        # evasion (simplified — keep same logic)
        score += self._evasion(group, initial_poses, final_poses,
                               initial_obstacle_states, final_obstacle_states)

        # stuckness
        score += self._stuckness(group, initial_poses, final_poses,
                                 n_new_reached > 0, reached_before)
        return score

    def _evasion(self, group, ip, fp, ios, fos):
        """Evasion reward — same as allrobotreach compute_evasion_reward."""
        k_align = 5.0
        k_dist = 3.0
        robot_r = 0.2
        obs_r = 0.7
        r_col = 2 * robot_r
        o_col = obs_r + robot_r
        r_prox = 1.5
        o_prox = CONFIG["obstacle_proximity_threshold"]
        r_align_th = 0.7
        o_align_th = 0.7
        ev = 0.0
        for i in group:
            xi0, yi0, th0 = ip[i]
            xi1, yi1, th1 = fp[i]
            # robot-robot
            for j in range(self.num_robots):
                if i == j:
                    continue
                dx0 = ip[j][0] - xi0; dy0 = ip[j][1] - yi0
                cd0 = np.sqrt(dx0**2 + dy0**2)
                cl0 = cd0 - r_col
                if cl0 > r_prox:
                    continue
                a2j0 = np.arctan2(dy0, dx0)
                al0 = np.cos(th0 - a2j0)
                dx1 = fp[j][0] - xi1; dy1 = fp[j][1] - yi1
                cd1 = np.sqrt(dx1**2 + dy1**2)
                cl1 = cd1 - r_col
                a2j1 = np.arctan2(dy1, dx1)
                al1 = np.cos(th1 - a2j1)
                cl1c = min(cl1, r_prox)
                ci = cl1c - cl0
                urg = max(0, r_prox - cl0) / r_prox
                ev += urg * k_dist * ci
                if cl0 <= r_align_th:
                    ai = al0 - al1
                    ua = max(0, r_align_th - cl0) / r_align_th
                    ev += ua * k_align * ai
            # robot-obstacle
            for oi in range(self.num_obstacles):
                ox0, oy0 = ios[oi, 0], ios[oi, 1]
                dx0 = ox0 - xi0; dy0 = oy0 - yi0
                cd0 = np.sqrt(dx0**2 + dy0**2)
                cl0 = cd0 - o_col
                if cl0 > o_prox:
                    continue
                a2o0 = np.arctan2(dy0, dx0)
                al0 = np.cos(th0 - a2o0)
                ox1, oy1 = fos[oi, 0], fos[oi, 1]
                dx1 = ox1 - xi1; dy1 = oy1 - yi1
                cd1 = np.sqrt(dx1**2 + dy1**2)
                cl1 = cd1 - o_col
                a2o1 = np.arctan2(dy1, dx1)
                al1 = np.cos(th1 - a2o1)
                cl1c = min(cl1, o_prox)
                ci = cl1c - cl0
                urg = max(0, o_prox - cl0) / o_prox
                ev += urg * k_dist * ci
                if cl0 <= o_align_th:
                    ai = al0 - al1
                    ua = max(0, o_align_th - cl0) / o_align_th
                    ev += ua * k_align * ai
        return ev

    def _stuckness(self, group, ip, fp, had_new_reach, reached):
        if had_new_reach:
            return 0.0
        k_stuck = 20.0
        min_disp = CONFIG["min_displacement_threshold"]
        min_rot = CONFIG["min_rotation_threshold"]
        unreached = [i for i in group if reached is None or not reached[i]]
        if not unreached:
            return 0.0
        td = 0.0; tr = 0.0
        for i in unreached:
            d = np.sqrt((fp[i][0] - ip[i][0])**2 + (fp[i][1] - ip[i][1])**2)
            td += d
            ad = abs(fp[i][2] - ip[i][2])
            ad = min(ad, 2 * np.pi - ad)
            tr += ad
        ad_ = td / len(unreached)
        ar_ = tr / len(unreached)
        if ad_ < min_disp and ar_ < min_rot:
            return -k_stuck * max(0, min_disp - ad_)
        return 0.0

    # ----- evaluate all groups (Phase 1) -----
    def evaluate_all_groups(self, poses, distance, cos_, sin_, collision,
                            action, goal_positions, obstacle_states,
                            reached, urgency_flags):
        """
        Run oracle evaluation for every candidate group.
        Returns list of scores (same length as self.groups).
        """
        snapshot = SimulationSnapshot.from_sim(self.sim)
        scores = []
        for group in self.groups:
            total = 0.0
            for _ in range(self.n_rollouts_per_group):
                s = self._evaluate_group_once(
                    group, poses, distance, cos_, sin_, collision, action,
                    goal_positions, obstacle_states, snapshot,
                    reached, urgency_flags,
                )
                total += s
            scores.append(total / self.n_rollouts_per_group)
        return scores


# =============================================================================
# Trajectory recorder
# =============================================================================
class TrajectoryRecorder:
    """
    Records full episode trajectories with oracle diagnostics.

    Saved data per episode:
      - initial_config: robot poses, goals, obstacle states at episode start
      - steps[]: per-step records (see _make_step_record)
      - outcome: "all_reached" | "timeout" | "collision" | "oob"
      - per_robot_reach_step: step at which each robot reached (or -1)
      - total_steps: number of Phase-2 sim steps in this episode
    """

    def __init__(self, oracle: OracleEvaluator, sim, policy, groups):
        self.oracle = oracle
        self.sim = sim
        self.policy = policy
        self.groups = groups
        self.num_robots = sim.num_robots
        self.num_obstacles = sim.num_obstacles

    def run_episodes(self, n_samples: int) -> Dict:
        """
        Run episodes collecting diagnostic trajectory data.

        The outer loop counts Phase-1/Phase-2 sample steps (like allrobotreach).
        We stop when n_samples steps have been taken and also finalize the
        current episode so every episode in the output is complete.
        """
        N = self.num_robots
        goal_threshold = CONFIG["goal_reach_threshold"]
        max_steps = CONFIG["max_steps_per_episode"]
        phase2_mode = CONFIG["phase2_selection"]
        phase2_temperature = CONFIG["phase2_temperature"]
        horizon = self.oracle.horizon
        urgency_lookback = CONFIG["urgency_lookback_window"]
        urgency_stuck_threshold = CONFIG["urgency_stuck_threshold"]

        all_episodes = []
        sample_count = 0
        episode_count = 0

        # ---- reset ----
        (poses, distance, cos_, sin_, collision, goals,
         action, reward, positions, goal_positions, obstacle_states
         ) = self.sim.reset()
        step_in_episode = 0
        reached = [False] * N
        robot_distance_history = [[] for _ in range(N)]
        urgency_flags = [False] * N
        per_robot_reach_step = [-1] * N

        # Record initial config
        initial_config = {
            "poses": [p.copy() for p in poses],
            "goal_positions": [g.copy() for g in goal_positions],
            "obstacle_states": obstacle_states.copy(),
            "distances": list(distance),
        }
        episode_steps = []

        pbar = tqdm(total=n_samples, desc="Recording trajectories")

        while sample_count < n_samples:
            # ---- urgency tracking (same as allrobotreach) ----
            for ri in range(N):
                if not reached[ri]:
                    robot_distance_history[ri].append(distance[ri])
                    if len(robot_distance_history[ri]) > urgency_lookback:
                        robot_distance_history[ri].pop(0)
                    if len(robot_distance_history[ri]) >= urgency_lookback:
                        prog = robot_distance_history[ri][0] - robot_distance_history[ri][-1]
                        urgency_flags[ri] = (prog < urgency_stuck_threshold)
                    else:
                        urgency_flags[ri] = False
                else:
                    urgency_flags[ri] = False
                    robot_distance_history[ri] = []

            # ---- Phase 1: oracle evaluation of all groups ----
            robot_state_obs, _ = self.policy.prepare_state(
                poses, distance, cos_, sin_, collision, action, goal_positions,
            )
            robot_obs = np.array(robot_state_obs)

            # Raw policy actions (before any coupling)
            raw_policy_actions = self.oracle.get_raw_policy_action(robot_obs, obstacle_states)
            # raw_policy_actions shape: (N, 2), values in [-1, 1]

            # Per-robot scaled linear velocities from raw policy
            per_robot_raw_lin_vel = [(raw_policy_actions[i][0] + 1) / 4 for i in range(N)]
            per_robot_raw_ang_vel = [raw_policy_actions[i][1] for i in range(N)]

            group_scores = self.oracle.evaluate_all_groups(
                poses, distance, cos_, sin_, collision, action,
                goal_positions, obstacle_states, reached, urgency_flags,
            )

            # ---- Phase 2: select group & advance ----
            scores_t = torch.tensor(group_scores, dtype=torch.float32)
            if phase2_mode == "softmax":
                logits = scores_t / phase2_temperature
                probs = torch.softmax(logits, dim=0)
                chosen_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                chosen_idx = random.randrange(len(self.groups))
                probs = torch.full((len(self.groups),), 1.0 / len(self.groups))

            selected_group = self.groups[chosen_idx]
            selected_score = group_scores[chosen_idx]

            # Snapshot state before Phase 2
            pre_poses = [p.copy() for p in poses]
            pre_distances = list(distance)

            # Execute Phase 2 (H steps, identical to allrobotreach)
            phase2_actions_log = []  # per-sub-step actions
            phase2_early_stop = False
            phase2_reason = None

            for ph2 in range(horizon):
                rs, _ = self.policy.prepare_state(
                    poses, distance, cos_, sin_, collision, action, goal_positions,
                )
                applied_action, raw_act = self.oracle.get_action_for_group(
                    np.array(rs), obstacle_states, selected_group,
                )

                phase2_actions_log.append({
                    "applied": [a.copy() for a in applied_action],
                    "raw_policy": raw_act.copy(),  # (N,2) before coupling
                })

                (poses, distance, cos_, sin_, collision, goals,
                 action, reward, positions, goal_positions, obstacle_states
                 ) = self.sim.step(applied_action, None, None)
                step_in_episode += 1

                # Update reached
                for ri in range(N):
                    if not reached[ri] and distance[ri] < goal_threshold:
                        reached[ri] = True
                        per_robot_reach_step[ri] = step_in_episode

                if any(collision):
                    phase2_early_stop = True
                    phase2_reason = "collision"
                    break
                if outside_of_bounds(poses, self.sim):
                    phase2_early_stop = True
                    phase2_reason = "oob"
                    break
                if all(reached):
                    phase2_early_stop = True
                    phase2_reason = "all_reached"
                    break
                if step_in_episode >= max_steps:
                    phase2_early_stop = True
                    phase2_reason = "timeout"
                    break

            # Post Phase 2 state
            post_poses = [p.copy() for p in poses]
            post_distances = list(distance)

            # Per-robot displacement during Phase 2
            per_robot_disp = []
            for ri in range(N):
                dx = post_poses[ri][0] - pre_poses[ri][0]
                dy = post_poses[ri][1] - pre_poses[ri][1]
                per_robot_disp.append(np.sqrt(dx**2 + dy**2))

            # Record step
            step_record = {
                "step_in_episode": step_in_episode,
                # Phase 1 — oracle scores
                "group_scores": group_scores,             # list[float], len=n_groups
                "best_group_idx": int(np.argmax(group_scores)),
                "best_group": self.groups[int(np.argmax(group_scores))],
                "best_score": float(np.max(group_scores)),
                "worst_score": float(np.min(group_scores)),
                "mean_score": float(np.mean(group_scores)),
                # Phase 2 — selection
                "selected_group_idx": chosen_idx,
                "selected_group": selected_group,
                "selected_score": selected_score,
                "selection_probs_top5": self._top5_probs(probs),
                # Per-robot raw policy output
                "raw_lin_vel": per_robot_raw_lin_vel,      # list[float] len=N
                "raw_ang_vel": per_robot_raw_ang_vel,      # list[float] len=N
                # State before Phase 2
                "pre_poses": pre_poses,
                "pre_distances": pre_distances,
                # State after Phase 2
                "post_poses": post_poses,
                "post_distances": post_distances,
                "per_robot_displacement": per_robot_disp,
                # Phase 2 sub-step actions
                "phase2_actions": phase2_actions_log,
                # Flags
                "reached": list(reached),
                "urgency_flags": list(urgency_flags),
                "collision": list(collision),
                "phase2_early_stop": phase2_early_stop,
                "phase2_reason": phase2_reason,
                "obstacle_states": obstacle_states.copy(),
            }
            episode_steps.append(step_record)

            sample_count += 1
            pbar.update(1)
            pbar.set_postfix({
                "ep": episode_count,
                "ep_step": step_in_episode,
                "reached": f"{sum(reached)}/{N}",
            })

            # ---- episode reset check ----
            all_reached = all(reached)
            should_reset = (
                any(collision) or
                step_in_episode >= max_steps or
                outside_of_bounds(poses, self.sim) or
                all_reached
            )

            if should_reset:
                if all_reached:
                    outcome = "all_reached"
                elif any(collision):
                    outcome = "collision"
                elif outside_of_bounds(poses, self.sim):
                    outcome = "oob"
                else:
                    outcome = "timeout"

                # Which robots never reached
                never_reached = [ri for ri in range(N) if not reached[ri]]

                episode_record = {
                    "episode_id": episode_count,
                    "initial_config": initial_config,
                    "steps": episode_steps,
                    "outcome": outcome,
                    "total_sim_steps": step_in_episode,
                    "per_robot_reach_step": list(per_robot_reach_step),
                    "never_reached_robots": never_reached,
                    "n_reached": sum(reached),
                }
                all_episodes.append(episode_record)
                episode_count += 1

                # Reset
                (poses, distance, cos_, sin_, collision, goals,
                 action, reward, positions, goal_positions, obstacle_states
                 ) = self.sim.reset(random_obstacles=True)
                step_in_episode = 0
                reached = [False] * N
                robot_distance_history = [[] for _ in range(N)]
                urgency_flags = [False] * N
                per_robot_reach_step = [-1] * N
                initial_config = {
                    "poses": [p.copy() for p in poses],
                    "goal_positions": [g.copy() for g in goal_positions],
                    "obstacle_states": obstacle_states.copy(),
                    "distances": list(distance),
                }
                episode_steps = []

        pbar.close()

        # If there's an unfinished episode, continue it to completion
        if episode_steps:
            # Run remaining steps until episode ends
            print(f"\nFinishing current episode (step {step_in_episode}/{max_steps})...")
            while True:
                # urgency
                for ri in range(N):
                    if not reached[ri]:
                        robot_distance_history[ri].append(distance[ri])
                        if len(robot_distance_history[ri]) > urgency_lookback:
                            robot_distance_history[ri].pop(0)
                        if len(robot_distance_history[ri]) >= urgency_lookback:
                            prog = robot_distance_history[ri][0] - robot_distance_history[ri][-1]
                            urgency_flags[ri] = (prog < urgency_stuck_threshold)
                        else:
                            urgency_flags[ri] = False
                    else:
                        urgency_flags[ri] = False
                        robot_distance_history[ri] = []

                rs_obs, _ = self.policy.prepare_state(
                    poses, distance, cos_, sin_, collision, action, goal_positions,
                )
                r_obs = np.array(rs_obs)
                raw_acts = self.oracle.get_raw_policy_action(r_obs, obstacle_states)
                per_robot_raw_lv = [(raw_acts[i][0] + 1) / 4 for i in range(N)]
                per_robot_raw_av = [raw_acts[i][1] for i in range(N)]

                g_scores = self.oracle.evaluate_all_groups(
                    poses, distance, cos_, sin_, collision, action,
                    goal_positions, obstacle_states, reached, urgency_flags,
                )
                scores_t2 = torch.tensor(g_scores, dtype=torch.float32)
                if phase2_mode == "softmax":
                    logits2 = scores_t2 / phase2_temperature
                    probs2 = torch.softmax(logits2, dim=0)
                    cidx2 = torch.multinomial(probs2, num_samples=1).item()
                else:
                    cidx2 = random.randrange(len(self.groups))
                    probs2 = torch.full((len(self.groups),), 1.0 / len(self.groups))

                sel_grp = self.groups[cidx2]
                pre_p = [p.copy() for p in poses]
                pre_d = list(distance)

                p2_acts = []
                p2_early = False
                p2_reason2 = None
                for ph2 in range(horizon):
                    rs2, _ = self.policy.prepare_state(
                        poses, distance, cos_, sin_, collision, action, goal_positions,
                    )
                    app_act, raw2 = self.oracle.get_action_for_group(
                        np.array(rs2), obstacle_states, sel_grp,
                    )
                    p2_acts.append({"applied": [a.copy() for a in app_act], "raw_policy": raw2.copy()})
                    (poses, distance, cos_, sin_, collision, goals,
                     action, reward, positions, goal_positions, obstacle_states
                     ) = self.sim.step(app_act, None, None)
                    step_in_episode += 1
                    for ri in range(N):
                        if not reached[ri] and distance[ri] < goal_threshold:
                            reached[ri] = True
                            per_robot_reach_step[ri] = step_in_episode
                    if any(collision):
                        p2_early = True; p2_reason2 = "collision"; break
                    if outside_of_bounds(poses, self.sim):
                        p2_early = True; p2_reason2 = "oob"; break
                    if all(reached):
                        p2_early = True; p2_reason2 = "all_reached"; break
                    if step_in_episode >= max_steps:
                        p2_early = True; p2_reason2 = "timeout"; break

                post_p = [p.copy() for p in poses]
                post_d = list(distance)
                disp = [np.sqrt((post_p[ri][0]-pre_p[ri][0])**2 +
                                (post_p[ri][1]-pre_p[ri][1])**2) for ri in range(N)]

                step_rec = {
                    "step_in_episode": step_in_episode,
                    "group_scores": g_scores,
                    "best_group_idx": int(np.argmax(g_scores)),
                    "best_group": self.groups[int(np.argmax(g_scores))],
                    "best_score": float(np.max(g_scores)),
                    "worst_score": float(np.min(g_scores)),
                    "mean_score": float(np.mean(g_scores)),
                    "selected_group_idx": cidx2,
                    "selected_group": sel_grp,
                    "selected_score": g_scores[cidx2],
                    "selection_probs_top5": self._top5_probs(probs2),
                    "raw_lin_vel": per_robot_raw_lv,
                    "raw_ang_vel": per_robot_raw_av,
                    "pre_poses": pre_p,
                    "pre_distances": pre_d,
                    "post_poses": post_p,
                    "post_distances": post_d,
                    "per_robot_displacement": disp,
                    "phase2_actions": p2_acts,
                    "reached": list(reached),
                    "urgency_flags": list(urgency_flags),
                    "collision": list(collision),
                    "phase2_early_stop": p2_early,
                    "phase2_reason": p2_reason2,
                    "obstacle_states": obstacle_states.copy(),
                }
                episode_steps.append(step_rec)

                all_r = all(reached)
                should_r = (any(collision) or step_in_episode >= max_steps or
                            outside_of_bounds(poses, self.sim) or all_r)
                if should_r:
                    if all_r:
                        oc = "all_reached"
                    elif any(collision):
                        oc = "collision"
                    elif outside_of_bounds(poses, self.sim):
                        oc = "oob"
                    else:
                        oc = "timeout"
                    nr = [ri for ri in range(N) if not reached[ri]]
                    all_episodes.append({
                        "episode_id": episode_count,
                        "initial_config": initial_config,
                        "steps": episode_steps,
                        "outcome": oc,
                        "total_sim_steps": step_in_episode,
                        "per_robot_reach_step": list(per_robot_reach_step),
                        "never_reached_robots": nr,
                        "n_reached": sum(reached),
                    })
                    episode_count += 1
                    break

        return {
            "episodes": all_episodes,
            "groups": self.groups,
            "config": dict(CONFIG),
            "timestamp": datetime.now().isoformat(),
        }

    def _top5_probs(self, probs):
        """Return top-5 (group, prob) tuples."""
        if probs is None:
            return []
        top5_vals, top5_idx = torch.topk(probs, min(5, len(probs)))
        return [(self.groups[j.item()], float(top5_vals[k])) for k, j in enumerate(top5_idx)]


# =============================================================================
# Summary statistics printed to console
# =============================================================================
def print_summary(data: Dict):
    episodes = data["episodes"]
    n_ep = len(episodes)
    outcomes = [e["outcome"] for e in episodes]
    n_success = outcomes.count("all_reached")
    n_timeout = outcomes.count("timeout")
    n_collision = outcomes.count("collision")
    n_oob = outcomes.count("oob")
    N = CONFIG["n_robots"]

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Episodes: {n_ep}")
    print(f"  all_reached: {n_success}  ({n_success/n_ep*100:.1f}%)")
    print(f"  timeout:     {n_timeout}  ({n_timeout/n_ep*100:.1f}%)")
    print(f"  collision:   {n_collision}  ({n_collision/n_ep*100:.1f}%)")
    print(f"  oob:         {n_oob}  ({n_oob/n_ep*100:.1f}%)")

    # ---- Timeout analysis ----
    timeout_eps = [e for e in episodes if e["outcome"] == "timeout"]
    if timeout_eps:
        print(f"\n--- TIMEOUT EPISODES ({len(timeout_eps)}) ---")
        reached_counts = [e["n_reached"] for e in timeout_eps]
        print(f"  Robots reached per timeout episode: "
              f"min={min(reached_counts)} max={max(reached_counts)} "
              f"mean={np.mean(reached_counts):.1f}")

        # Per-robot starvation analysis: how often is each robot the one stuck
        stuck_counts = np.zeros(N, dtype=int)
        for e in timeout_eps:
            for ri in e["never_reached_robots"]:
                stuck_counts[ri] += 1
        print(f"\n  Per-robot stuck frequency (in timeout episodes):")
        for ri in range(N):
            bar = "#" * stuck_counts[ri]
            print(f"    Robot {ri:>2}: {stuck_counts[ri]:>3}/{len(timeout_eps)}  {bar}")

        # Velocity analysis: are stuck robots getting zero-velocity from policy?
        print(f"\n  Stuck robot velocity analysis (last 20 steps of timeout episodes):")
        for ep in timeout_eps[:5]:  # show first 5
            stuck = ep["never_reached_robots"]
            if not stuck:
                continue
            last_steps = ep["steps"][-20:]
            print(f"\n  Episode {ep['episode_id']} — stuck robots: {stuck}, "
                  f"reached {ep['n_reached']}/{N}, steps={ep['total_sim_steps']}")
            for ri in stuck[:3]:  # show first 3 stuck robots
                lv_vals = [s["raw_lin_vel"][ri] for s in last_steps]
                av_vals = [s["raw_ang_vel"][ri] for s in last_steps]
                disp_vals = [s["per_robot_displacement"][ri] for s in last_steps]
                # How often was this robot in the selected group?
                n_selected = sum(1 for s in last_steps if ri in s["selected_group"])
                print(f"    Robot {ri}: avg_lin_vel={np.mean(lv_vals):.4f} "
                      f"avg_ang_vel={np.mean(av_vals):.4f} "
                      f"avg_disp={np.mean(disp_vals):.4f} "
                      f"selected_in={n_selected}/{len(last_steps)} steps")

        # Group selection patterns for stuck robots
        print(f"\n  Group selection frequency for stuck robots (all timeout eps):")
        # How often are single-robot groups of stuck robots selected?
        for ep in timeout_eps:
            stuck = ep["never_reached_robots"]
            if not stuck:
                continue
            total_steps = len(ep["steps"])
            for ri in stuck:
                n_single = sum(1 for s in ep["steps"] if s["selected_group"] == [ri])
                n_any = sum(1 for s in ep["steps"] if ri in s["selected_group"])
                # urgency: how many steps had urgency flag
                n_urgent = sum(1 for s in ep["steps"] if s["urgency_flags"][ri])
                print(f"    Ep{ep['episode_id']} Robot{ri}: "
                      f"in_selected={n_any}/{total_steps} "
                      f"as_singleton={n_single}/{total_steps} "
                      f"urgent_steps={n_urgent}/{total_steps} "
                      f"final_dist={ep['steps'][-1]['post_distances'][ri]:.2f}")

    # ---- Success episode stats ----
    success_eps = [e for e in episodes if e["outcome"] == "all_reached"]
    if success_eps:
        steps_list = [e["total_sim_steps"] for e in success_eps]
        print(f"\n--- SUCCESSFUL EPISODES ({len(success_eps)}) ---")
        print(f"  Steps: min={min(steps_list)} max={max(steps_list)} "
              f"mean={np.mean(steps_list):.0f} median={np.median(steps_list):.0f}")

    # ---- Zero-velocity analysis across ALL episodes ----
    print(f"\n--- POLICY ZERO-VELOCITY ANALYSIS (all episodes) ---")
    all_steps_flat = []
    for e in episodes:
        all_steps_flat.extend(e["steps"])
    if all_steps_flat:
        # Per robot: fraction of steps where raw policy outputs near-zero linear vel
        zero_thresh = 0.02  # scaled linear vel < this ≈ stopped
        for ri in range(N):
            n_near_zero = sum(1 for s in all_steps_flat if s["raw_lin_vel"][ri] < zero_thresh)
            frac = n_near_zero / len(all_steps_flat)
            bar = "#" * int(frac * 50)
            print(f"  Robot {ri:>2}: near-zero lin_vel in {n_near_zero:>5}/{len(all_steps_flat)} "
                  f"steps ({frac*100:>5.1f}%)  {bar}")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================
def main():
    from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
    from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

    config = CONFIG
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("Diagnostic Trajectory Recording")
    print("=" * 70)
    print(f"Samples: {config['n_samples']}")
    print(f"Robots: {config['n_robots']}, Obstacles: {config['n_obstacles']}")
    print(f"Max steps/episode: {config['max_steps_per_episode']}")
    print(f"Oracle horizon: {config['oracle_horizon']}")
    print(f"Phase 2: {config['phase2_selection']} (temp={config['phase2_temperature']})")
    print(f"Output: {config['output_path']}")
    print("=" * 70 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    sim = MARL_SIM_OBSTACLE(
        world_file=config["world_file"],
        disable_plotting=config["disable_plotting"],
        reward_phase=5,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=config["obstacle_proximity_threshold"],
    )
    logger.info(f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles")

    policy = TD3Obstacle(
        state_dim=config["state_dim"],
        action_dim=2,
        max_action=1.0,
        device=device,
        num_robots=config["n_robots"],
        num_obstacles=config["n_obstacles"],
        obstacle_state_dim=config["obstacle_state_dim"],
        load_model=True,
        model_name=config["decentralized_model_name"],
        load_model_name=config["decentralized_model_name"],
        load_directory=Path(config["decentralized_model_directory"]),
        save_directory=Path(config["decentralized_model_directory"]),
    )
    logger.info("Policy loaded")

    groups = generate_candidate_groups(
        num_robots=config["n_robots"],
        include_size_1=config["include_size_1"],
        include_size_2=config["include_size_2"],
        include_size_3=config["include_size_3"],
        include_size_4=config.get("include_size_4", False),
        include_size_7=config.get("include_size_7", False),
    )
    logger.info(f"Groups: {len(groups)} total "
                f"(size-1:{sum(1 for g in groups if len(g)==1)} "
                f"size-2:{sum(1 for g in groups if len(g)==2)} "
                f"size-3:{sum(1 for g in groups if len(g)==3)} "
                f"size-4:{sum(1 for g in groups if len(g)==4)} "
                f"size-7:{sum(1 for g in groups if len(g)==7)})")

    oracle = OracleEvaluator(
        sim=sim, policy=policy, groups=groups,
        horizon=config["oracle_horizon"],
        n_rollouts_per_group=config["n_rollouts_per_group"],
        device=device,
    )

    recorder = TrajectoryRecorder(oracle, sim, policy, groups)
    data = recorder.run_episodes(config["n_samples"])

    # Save
    out = Path(config["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f"\nSaved {len(data['episodes'])} episodes to {out}")

    # Print summary
    print_summary(data)

    # Print data format
    n_ep = len(data["episodes"])
    print(f"\nData format:")
    print(f"  data['episodes']: list of {n_ep} episode dicts")
    print(f"  episode['initial_config']: poses, goal_positions, obstacle_states, distances")
    print(f"  episode['steps'][t]: per-step dict with:")
    print(f"    group_scores, selected_group, raw_lin_vel, raw_ang_vel,")
    print(f"    pre_poses, post_poses, pre_distances, post_distances,")
    print(f"    per_robot_displacement, phase2_actions, reached, urgency_flags, ...")
    print(f"  episode['outcome']: 'all_reached' | 'timeout' | 'collision' | 'oob'")
    print(f"  episode['per_robot_reach_step']: step each robot reached (-1 = never)")
    print(f"  episode['never_reached_robots']: list of robot indices that never reached")
    print(f"\nTo replay: load data, set sim states from initial_config, apply")
    print(f"  phase2_actions at each step to reproduce the trajectory exactly.")


if __name__ == "__main__":
    main()
