"""
Switcher Environment — wraps MARL_SIM_OBSTACLE at the group-selection level.

One ``SwitcherEnv.step(group_idx)`` call:
  1. Converts the selected group into coupled robot actions
     (via the frozen decentralized TD3Obstacle policy).
  2. Runs ``selection_interval`` low-level simulation steps.
  3. Computes a **swarm-level** reward for that interval.
  4. Builds the next observation (group features + state features) via
     ``RLFeatureBuilder``.

The agent (PPO) only sees the coarse time-scale: one decision every
``selection_interval`` sim steps.

Swarm-level reward per interval
-------------------------------
  r = r_progress + r_reach + r_sync + r_evasion + r_collision + r_time

  r_progress   k_p · Σ_i (d_i^before − d_i^after)   (summed over active group)
  r_reach      k_r / n_remaining  per newly-reached robot
  r_sync       k_s · (var_before − var_after)        (variance of dist_to_goal)
  r_evasion    k_e · evasion_score                   (clearance + alignment improvement)
  r_collision  −200 if any collision during interval  (episode terminates)
  r_time       −c   per interval                     (time pressure)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.groups.group_generator import (
    generate_all_groups,
    filter_groups_by_size,
)
from robot_nav.models.MARL.groups.action_coupling import actions_for_group_from_raw
from robot_nav.models.MARL.switcher.rl.rl_feature_builder import RLFeatureBuilder
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE


# ──────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────
def _outside_of_bounds(poses, sim) -> bool:
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def _generate_groups(
    num_robots: int,
    include_sizes: Tuple[int, ...] = (1, 2, 3, 4, 7),
) -> List[List[int]]:
    """Generate candidate groups via binary allocation."""
    m = 3 if num_robots <= 6 else 4
    all_groups = generate_all_groups(m=m, n=num_robots, use_complement=True)
    allowed = set(include_sizes)
    return [g for g in all_groups if len(g) in allowed]


# ──────────────────────────────────────────────────────────────────────
# Switcher Environment
# ──────────────────────────────────────────────────────────────────────
class SwitcherEnv:
    debug_rewards: bool = False  # Set True to print reward breakdowns
    """
    Gym-like wrapper that exposes group selection as the action space.

    Observation = ``(group_features, state_features)`` from RLFeatureBuilder.
    Action      = integer index into ``self.groups``.
    Reward      = swarm-level scalar accumulated over ``selection_interval`` steps.
    Done        = collision / out-of-bounds / all-reached / timeout.

    Args:
        sim: Pre-created MARL_SIM_OBSTACLE instance.
        policy: Frozen TD3Obstacle decentralized policy.
        groups: Candidate groups (list of robot-index lists).
        feature_builder: ``RLFeatureBuilder`` instance.
        selection_interval: Number of low-level sim steps per switcher decision.
        max_episode_steps: Episode horizon in *sim* steps (not decisions).
        goal_threshold: Distance to consider a robot "reached".
        use_rotation_coupling: If True, groups of size > 3 get coupled
            angular velocity (average) in addition to coupled linear.
        rotation_coupling_threshold: Group sizes strictly above this get
            rotation coupling. Default 3.
        device: Torch device for feature computation.

    Reward coefficients (class attributes — override per-instance if needed):
        k_progress: Weight for distance-progress reward.
        k_reach: Bonus weight for each newly-reached robot.
        k_all_reached: Large bonus when ALL robots reach their goals.
        k_sync: Weight for variance-reduction (synchronisation) reward.
        k_evasion: Weight for evasion/proximity reward (clearance + alignment).
        collision_penalty: Flat penalty on collision (episode ends).
        time_penalty: Small per-interval cost (encourages efficiency).
        robot_proximity_threshold: Clearance threshold for robot-robot evasion.
        obstacle_proximity_threshold: Clearance threshold for robot-obstacle evasion.
    """

    # Default reward coefficients (tunable)
    k_progress: float = 3.0
    k_reach: float = 50.0
    k_all_reached: float = 500.0        # large bonus when ALL robots reach goals
    k_sync: float = 8.0
    k_evasion: float = 1.0
    collision_penalty: float = -200.0
    time_penalty: float = -0.1

    # Evasion geometry constants
    robot_radius: float = 0.2
    obstacle_radius: float = 0.7
    robot_proximity_threshold: float = 1.25
    obstacle_proximity_threshold: float = 1.25

    def __init__(
        self,
        sim: MARL_SIM_OBSTACLE,
        policy: TD3Obstacle,
        groups: List[List[int]],
        feature_builder: RLFeatureBuilder,
        selection_interval: int = 10,
        max_episode_steps: int = 1000,
        goal_threshold: float = 0.3,
        use_rotation_coupling: bool = True,
        rotation_coupling_threshold: int = 3,
        device: str = "cpu",
    ):
        self.sim = sim
        self.policy = policy
        self.groups = groups
        self.fb = feature_builder
        self.selection_interval = selection_interval
        self.max_episode_steps = max_episode_steps
        self.goal_threshold = goal_threshold
        self.use_rotation_coupling = use_rotation_coupling
        self.rotation_coupling_threshold = rotation_coupling_threshold
        self.device = torch.device(device)

        self.num_robots: int = sim.num_robots
        self.num_groups: int = len(groups)

        # Pre-compute group sets for fast membership lookup in _actions_for_group
        self._group_sets = [set(g) for g in groups]

        # Cached obstacle tensor on GPU (static obstacles — reuse across steps).
        # Updated on reset() when obstacles are re-randomised.
        self._obstacle_t: Optional[torch.Tensor] = None

        # Episode state (set by reset())
        self._poses = None
        self._distance = None
        self._cos = None
        self._sin = None
        self._collision = None
        self._action = None
        self._goal_positions = None
        self._obstacle_states = None
        self._reached: List[bool] = []
        self._step_count: int = 0

    # ──────────────────────── reset ────────────────────────
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset environment and return initial observation.

        Returns:
            group_features: ``(M, D)`` tensor.
            state_features: ``(S,)`` tensor.
        """
        (
            self._poses, self._distance, self._cos, self._sin,
            self._collision, _, self._action, _,
            _, self._goal_positions, self._obstacle_states,
        ) = self.sim.reset(random_obstacles=True)

        self._reached = [False] * self.num_robots
        self._step_count = 0

        # Invalidate caches
        self._cached_h = None
        self._cached_attn_rr = None
        self._cached_attn_ro = None

        # Pre-compute obstacle tensor on GPU (static within the episode)
        self._obstacle_t = torch.as_tensor(
            np.array(self._obstacle_states),
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)  # (1, N_obs, obs_dim)

        return self._build_obs()

    # ──────────────────────── step ─────────────────────────
    def step(
        self, group_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool, dict]:
        """
        Execute one switcher decision for ``selection_interval`` sim steps.

        Args:
            group_idx: Index into ``self.groups``.

        Returns:
            group_features: ``(M, D)`` next observation.
            state_features: ``(S,)`` next observation.
            reward: Swarm-level scalar reward for this interval.
            done: Whether episode ended.
            info: Dict with episode-level diagnostics.
        """
        group = self.groups[group_idx]

        # Snapshot before interval
        dist_before = list(self._distance)
        reached_before = list(self._reached)
        n_reached_before = sum(reached_before)
        poses_before = np.asarray(self._poses, dtype=np.float32)      # (N, 3)
        obstacle_states_before = np.asarray(self._obstacle_states, dtype=np.float32)

        interval_collision = False
        interval_oob = False
        n_new_reached = 0

        # Cache for the last inner step's embeddings (avoids duplicate
        # attention forward pass in _build_obs after the loop).
        self._cached_h = None
        self._cached_attn_rr = None
        self._cached_attn_ro = None

        # Wrap entire inner sim loop in no_grad — all forward passes here
        # are inference-only (frozen policy)
        with torch.no_grad():
            for inner_step in range(self.selection_interval):
                # Build observation for the low-level policy
                robot_state, _ = self.policy.prepare_state(
                    self._poses, self._distance, self._cos, self._sin,
                    self._collision, self._action, self._goal_positions,
                )
                robot_obs = np.array(robot_state)

                # Single forward pass → actions + embeddings
                # We always cache the latest embeddings; the final cached
                # values will be from the pre-step state of the last inner
                # iteration.  After sim.step the state changes, so
                # _build_obs must still recompute (post-step state differs).
                raw_actions, h, attn_rr, attn_ro = (
                    self._get_actions_and_embeddings(robot_obs, self._obstacle_states)
                )
                self._cached_h = h
                self._cached_attn_rr = attn_rr
                self._cached_attn_ro = attn_ro

                action_out = actions_for_group_from_raw(
                    raw_actions, group, self.num_robots,
                    use_rotation_coupling=self.use_rotation_coupling,
                    rotation_coupling_threshold=self.rotation_coupling_threshold,
                )

                # Low-level sim step
                (
                    self._poses, self._distance, self._cos, self._sin,
                    self._collision, goals, self._action, _rewards,
                    _, self._goal_positions, self._obstacle_states,
                ) = self.sim.step(action_out, None, None)

                self._step_count += 1

                # Update sticky reached flags
                for i in range(self.num_robots):
                    if not self._reached[i] and self._distance[i] < self.goal_threshold:
                        self._reached[i] = True
                        n_new_reached += 1

                # Check termination within interval
                if any(self._collision):
                    interval_collision = True
                    break
                if _outside_of_bounds(self._poses, self.sim):
                    interval_oob = True
                    break
                if all(self._reached):
                    break
                if self._step_count >= self.max_episode_steps:
                    break

        # ── Compute swarm-level reward ──
        poses_after = np.asarray(self._poses, dtype=np.float32)         # (N, 3)
        obstacle_states_after = np.asarray(self._obstacle_states, dtype=np.float32)

        reward = self._compute_reward(
            group=group,
            dist_before=dist_before,
            dist_after=list(self._distance),
            reached_before=reached_before,
            n_new_reached=n_new_reached,
            n_reached_before=n_reached_before,
            had_collision=interval_collision or interval_oob,
            poses_before=poses_before,
            poses_after=poses_after,
            obstacle_states_before=obstacle_states_before,
            obstacle_states_after=obstacle_states_after,
        )

        # ── Done? ──
        all_reached = all(self._reached)
        timeout = self._step_count >= self.max_episode_steps
        done = interval_collision or interval_oob or all_reached or timeout

        # Give a final bonus / penalty on episode end
        if done and all_reached:
            # Large bonus for finishing — scaled inversely with time used
            frac_time_left = 1.0 - self._step_count / self.max_episode_steps
            reward += self.k_all_reached * (1.0 + frac_time_left)
        if done and timeout and not all_reached:
            reward -= 20.0  # timeout penalty

        # ── Observation ──
        group_features, state_features = self._build_obs()

        info = {
            "all_reached": all_reached,
            "collision": interval_collision,
            "oob": interval_oob,
            "timeout": timeout,
            "n_reached": sum(self._reached),
            "step_count": self._step_count,
            "n_new_reached": n_new_reached,
        }

        return group_features, state_features, reward, done, info

    # ──────────────────── reward ──────────────────────────
    def _compute_reward(
        self,
        group: List[int],
        dist_before: List[float],
        dist_after: List[float],
        reached_before: List[bool],
        n_new_reached: int,
        n_reached_before: int,
        had_collision: bool,
        poses_before: List[List[float]] = None,
        poses_after: List[List[float]] = None,
        obstacle_states_before: np.ndarray = None,
        obstacle_states_after: np.ndarray = None,
    ) -> float:
        """Compute swarm-level reward for one interval."""

        N = self.num_robots

        # Extra reward for large group size
        group_size_bonus = 5.0 if len(group) > 3 else 0.0

        # 1. Collision → big negative, nothing else matters
        if had_collision:
            if self.debug_rewards:
                print("[REWARD DEBUG] Collision: penalty=", self.collision_penalty)
                input("Press Enter to continue...")
            return self.collision_penalty

        reward = group_size_bonus
        reach_bonus = 0.0
        progress_reward = 0.0
        sync_reward = 0.0
        evasion_reward = 0.0
        time_penalty = self.time_penalty

        # 2. New-reach bonus: k_reach / n_remaining per newly-reached robot
        if n_new_reached > 0:
            n_remaining = N - n_reached_before
            for r in range(n_new_reached):
                rem = n_remaining - r
                if rem > 0:
                    reach_bonus += self.k_reach / rem
        reward += reach_bonus

        # 3. Progress reward — summed over *group members* that are unreached
        # for i in group:
        #     if reached_before[i]:
        #         continue
        #     progress = dist_before[i] - dist_after[i]
        #     progress_reward += self.k_progress * progress
        # reward += progress_reward

        # 4. Synchronisation: variance reduction across ALL robots (disabled for debugging)
        # if N >= 2:
        #     var_before = float(np.var(dist_before))
        #     var_after = float(np.var(dist_after))
        #     sync_reward = self.k_sync * (var_before - var_after)
        #     reward += sync_reward

        # 5. Evasion reward: encourage clearance improvement & turning away
        # if (
        #     self.k_evasion > 0
        #     and poses_before is not None
        #     and poses_after is not None
        # ):
        #     evasion = self._compute_evasion(
        #         group, poses_before, poses_after,
        #         obstacle_states_before, obstacle_states_after,
        #     )
        #     evasion_reward = self.k_evasion * evasion
        #     reward += evasion_reward

        # 6. Time penalty
        reward += time_penalty

        if self.debug_rewards:
            print(f"[REWARD DEBUG] reach_bonus={reach_bonus:.2f} progress={progress_reward:.2f} sync={sync_reward:.2f} evasion={evasion_reward:.2f} time_penalty={time_penalty:.2f} total={reward:.2f}")
            input("Press Enter to continue...")

        return reward

    # ──────────────────── evasion reward ──────────────────
    def _compute_evasion(
        self,
        group: List[int],
        poses_before,
        poses_after,
        obstacle_states_before: np.ndarray,
        obstacle_states_after: np.ndarray,
    ) -> float:
        """
        Compute evasion reward — rewards group robots for increasing clearance
        and turning away from nearby robots/obstacles.

        Fully vectorized with numpy broadcasting: replaces the original
        O(g × N × N_obs) nested Python loops with O(1) numpy operations.

        Returns:
            evasion_score: Positive = good evasive maneuver, negative = got closer.
        """
        if not group:
            return 0.0

        robot_coll_d = 2.0 * self.robot_radius            # 0.4
        obs_coll_d   = self.obstacle_radius + self.robot_radius  # 0.9
        k_align      = 5.0
        k_dist       = 3.0
        align_thresh = 0.7

        pb = np.asarray(poses_before, dtype=np.float32)   # (N, 3)
        pa = np.asarray(poses_after,  dtype=np.float32)   # (N, 3)
        g  = np.asarray(group, dtype=np.int64)            # (Ng,)
        Ng = len(g)

        xyi0 = pb[g, :2]    # (Ng, 2)  group positions before
        xyi1 = pa[g, :2]    # (Ng, 2)  group positions after
        thi0 = pb[g, 2:3]   # (Ng, 1)  group headings before
        thi1 = pa[g, 2:3]   # (Ng, 1)  group headings after

        evasion_score = 0.0

        # ── Robot-Robot evasion ─────────────────────────────────────────────
        # d0/d1: direction vectors from group robot i toward every robot j
        d0_rr = pb[:, :2][np.newaxis] - xyi0[:, np.newaxis]   # (Ng, N, 2)
        d1_rr = pa[:, :2][np.newaxis] - xyi1[:, np.newaxis]   # (Ng, N, 2)

        dist0_rr = np.sqrt((d0_rr ** 2).sum(-1))               # (Ng, N)
        dist1_rr = np.sqrt((d1_rr ** 2).sum(-1))
        cl0_rr   = dist0_rr - robot_coll_d                     # (Ng, N)
        cl1_rr   = dist1_rr - robot_coll_d

        # Exclude self-pairs
        self_mask = np.zeros((Ng, self.num_robots), dtype=bool)
        self_mask[np.arange(Ng), g] = True
        nearby_rr = (cl0_rr < self.robot_proximity_threshold) & ~self_mask  # (Ng, N)

        if nearby_rr.any():
            urgency_rr  = np.clip(self.robot_proximity_threshold - cl0_rr, 0.0, None) / self.robot_proximity_threshold
            improve_rr  = np.minimum(cl1_rr, self.robot_proximity_threshold) - cl0_rr
            evasion_score += float((urgency_rr * k_dist * improve_rr * nearby_rr).sum())

            close_rr = nearby_rr & (cl0_rr <= align_thresh)
            if close_rr.any():
                ang0_rr   = np.arctan2(d0_rr[..., 1], d0_rr[..., 0])   # (Ng, N)
                ang1_rr   = np.arctan2(d1_rr[..., 1], d1_rr[..., 0])
                align0_rr = np.cos(thi0 - ang0_rr)                      # (Ng, N)
                align1_rr = np.cos(thi1 - ang1_rr)
                urg_a_rr  = np.clip(align_thresh - cl0_rr, 0.0, None) / align_thresh
                evasion_score += float((urg_a_rr * k_align * (align0_rr - align1_rr) * close_rr).sum())

        # ── Robot-Obstacle evasion ───────────────────────────────────────────
        if obstacle_states_before is not None and len(obstacle_states_before) > 0:
            oxy0 = obstacle_states_before[:, :2]                        # (Nobs, 2)
            oxy1 = obstacle_states_after[:, :2]

            d0_ro  = oxy0[np.newaxis] - xyi0[:, np.newaxis]             # (Ng, Nobs, 2)
            d1_ro  = oxy1[np.newaxis] - xyi1[:, np.newaxis]
            dist0_ro = np.sqrt((d0_ro ** 2).sum(-1))                    # (Ng, Nobs)
            dist1_ro = np.sqrt((d1_ro ** 2).sum(-1))
            cl0_ro   = dist0_ro - obs_coll_d
            cl1_ro   = dist1_ro - obs_coll_d
            nearby_ro = cl0_ro < self.obstacle_proximity_threshold      # (Ng, Nobs)

            if nearby_ro.any():
                urgency_ro  = np.clip(self.obstacle_proximity_threshold - cl0_ro, 0.0, None) / self.obstacle_proximity_threshold
                improve_ro  = np.minimum(cl1_ro, self.obstacle_proximity_threshold) - cl0_ro
                evasion_score += float((urgency_ro * k_dist * improve_ro * nearby_ro).sum())

                close_ro = nearby_ro & (cl0_ro <= align_thresh)
                if close_ro.any():
                    ang0_ro   = np.arctan2(d0_ro[..., 1], d0_ro[..., 0])   # (Ng, Nobs)
                    ang1_ro   = np.arctan2(d1_ro[..., 1], d1_ro[..., 0])
                    align0_ro = np.cos(thi0 - ang0_ro)                      # (Ng, Nobs)
                    align1_ro = np.cos(thi1 - ang1_ro)
                    urg_a_ro  = np.clip(align_thresh - cl0_ro, 0.0, None) / align_thresh
                    evasion_score += float((urg_a_ro * k_align * (align0_ro - align1_ro) * close_ro).sum())

        return evasion_score


    # ──────────────────── observation ─────────────────────
    def _build_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (group_features, state_features) from current sim state.

        Uses cached embeddings from the last inner-loop forward pass when
        available.  After sim.step the robot state has changed by one
        time-step, so the cached embeddings are slightly stale — but the
        physical change within a single 100 ms step is tiny and the
        approximation avoids an expensive attention forward pass (~6 ms).
        On the very first call (after ``reset()``) there is no cache, so
        we fall back to a full ``_get_embeddings`` call.
        """
        if self._cached_h is not None:
            h = self._cached_h
            attn_rr = self._cached_attn_rr
            attn_ro = self._cached_attn_ro
        else:
            # First call after reset — no cache yet
            robot_state, _ = self.policy.prepare_state(
                self._poses, self._distance, self._cos, self._sin,
                self._collision, self._action, self._goal_positions,
            )
            robot_obs = np.array(robot_state)
            h, attn_rr, attn_ro = self._get_embeddings(robot_obs, self._obstacle_states)

        # Extra per-robot features (cheap, ~3 ms — always from fresh state)
        extra = self._get_extra_features()

        # Group features for actor
        group_features = self.fb(
            h, self.groups,
            h_glob=None,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            extra=extra,
        ).to(self.device)

        # State features for critic
        state_features = self.fb.build_state_features(
            h, h_glob=None, extra=extra,
        ).to(self.device)

        return group_features, state_features

    def _get_extra_features(self) -> Dict[str, torch.Tensor]:
        """Build the extra dict expected by RLFeatureBuilder."""
        N = self.num_robots
        dev = self.device

        dist_np = np.asarray(self._distance, dtype=np.float32)
        dist_to_goal = torch.as_tensor(dist_np, dtype=torch.float32, device=dev)

        # Vectorized clearance computation (single cdist call, not 14 separate ones)
        all_clearances = self.sim.get_robot_obstacle_clearances()  # (N, N_obs)
        if all_clearances.size > 0:
            min_clearances = all_clearances.min(axis=1)  # (N,)
        else:
            min_clearances = np.full(N, float('inf'), dtype=np.float32)
        clearance = torch.as_tensor(min_clearances, dtype=torch.float32, device=dev)

        reached_np = np.array(self._reached, dtype=np.float32)
        reached = torch.as_tensor(reached_np, dtype=torch.float32, device=dev)

        # Heading error: |atan2(sin, cos)| per robot — vectorized
        sin_arr = np.asarray(self._sin, dtype=np.float32)
        cos_arr = np.asarray(self._cos, dtype=np.float32)
        heading_np = np.abs(np.arctan2(sin_arr, cos_arr))
        heading_error = torch.as_tensor(heading_np, dtype=torch.float32, device=dev)

        frac_reached = float(reached_np.sum()) / N
        frac_reached_global = torch.full((N,), frac_reached, dtype=torch.float32, device=dev)

        unreached_mask = ~np.asarray(self._reached, dtype=bool)
        unreached_dists = dist_np[unreached_mask]
        var_val = float(np.var(unreached_dists)) if len(unreached_dists) >= 2 else 0.0
        var_dist_to_goal = torch.full((N,), var_val, dtype=torch.float32, device=dev)

        steps_frac = self._step_count / max(self.max_episode_steps, 1)
        steps_elapsed_frac = torch.full((N,), steps_frac, dtype=torch.float32, device=dev)

        return {
            "dist_to_goal": dist_to_goal,
            "clearance": clearance,
            "reached": reached,
            "heading_error": heading_error,
            "frac_reached_global": frac_reached_global,
            "var_dist_to_goal": var_dist_to_goal,
            "steps_elapsed_frac": steps_elapsed_frac,
        }

    # ──────────────────── policy helpers ──────────────────
    def _get_raw_actions(
        self, robot_obs: np.ndarray, obstacle_obs: np.ndarray,
    ) -> np.ndarray:
        """One GPU forward pass → (N, 2) raw actions in [-1, 1]."""
        action, _ = self.policy.get_action(robot_obs, obstacle_obs, add_noise=False)
        return action

    def _get_actions_and_embeddings(
        self, robot_obs: np.ndarray, obstacle_obs: np.ndarray,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single forward pass → actions AND attention embeddings.

        Avoids running the attention module twice (once for actions in
        the inner loop, once for observation building in ``_build_obs``).

        Returns:
            raw_actions: ``(N, 2)`` numpy array on CPU.
            h: ``(N, embed_dim)`` tensor on device (detached).
            attn_rr: ``(N, N)`` tensor on device (detached).
            attn_ro: ``(N, N_obs)`` tensor on device (detached).
        """
        robot_t = torch.as_tensor(
            robot_obs, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        # Reuse cached obstacle tensor (static within episode)
        obstacle_t = self._obstacle_t

        with torch.no_grad():
            (
                H, _, _, _, _, _,
                hard_weights_rr, hard_weights_ro, _,
            ) = self.policy.actor.attention(robot_t, obstacle_t)

            # Run policy head on the attention output to get actions
            action = self.policy.actor.policy_head(H)

        N = robot_t.shape[1]
        embed_dim = H.shape[-1]
        h = H.view(1, N, embed_dim).squeeze(0)             # (N, embed_dim)
        attn_rr = hard_weights_rr.squeeze(0)               # (N, N)
        attn_ro = hard_weights_ro.squeeze(0)                # (N, N_obs)

        raw_actions = action.cpu().data.numpy().reshape(-1, 2)
        return raw_actions, h, attn_rr, attn_ro

    def _get_embeddings(
        self, robot_obs: np.ndarray, obstacle_obs: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get h, attn_rr, attn_ro from the frozen policy's attention module."""
        robot_t = torch.as_tensor(
            robot_obs, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        # Reuse cached obstacle tensor when available (static within episode)
        if self._obstacle_t is not None:
            obstacle_t = self._obstacle_t
        else:
            obstacle_t = torch.as_tensor(
                np.array(obstacle_obs), dtype=torch.float32, device=self.device,
            ).unsqueeze(0)

        with torch.no_grad():
            (
                H, _, _, _, _, _,
                hard_weights_rr, hard_weights_ro, _,
            ) = self.policy.actor.attention(robot_t, obstacle_t)

        N = robot_t.shape[1]
        embed_dim = H.shape[-1]
        h = H.view(1, N, embed_dim).squeeze(0)             # (N, embed_dim)
        attn_rr = hard_weights_rr.squeeze(0)               # (N, N)
        attn_ro = hard_weights_ro.squeeze(0)                # (N, N_obs)
        return h, attn_rr, attn_ro

    # Action coupling is now handled by:
    #   robot_nav.models.MARL.groups.action_coupling.actions_for_group_from_raw
