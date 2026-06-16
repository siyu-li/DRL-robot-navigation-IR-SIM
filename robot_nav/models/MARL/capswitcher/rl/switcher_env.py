"""
SwitcherEnv: Gym-like environment wrapper for CAPSwitcher PPO training.

At each step the switcher commits to one mode for ``selection_interval``
simulation steps, then re-observes.

Observation space
-----------------
(512,) float32 — mean-pooled ``_pre_decoder_embedding`` over all robots.

Action space
------------
Discrete(2):  0 = coarse,  1 = precise.

Reward
------
Sum over sub-steps of:
    mean(per-robot simulation reward from phase 6)  +  mode_cost

where mode_cost = −0.2 (coarse) or −1.0 (precise) per sub-step.

Episode termination
-------------------
- any robot collision
- all robots reached their goals
- step count reaches max_steps
- any robot outside world bounds
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_nav.models.MARL.capswitcher.rl.reward import MODE_COSTS


def _outside_bounds(poses: list, sim) -> bool:
    """Return True if any robot pose is outside the world boundary."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


class SwitcherEnv:
    """
    Environment wrapper for training the CAPSwitcher binary switcher.

    Args:
        sim:                MARL_SIM_OBSTACLE instance (already initialised).
        backbone:           GATBackbone (frozen) — provides embeddings and
                            precise actions.
        coarse_steering:    CoarseSteering instance.
        selection_interval: Number of sim steps committed to one mode per
                            switcher decision. Default 5.
        max_steps:          Maximum sim steps per episode. Default 300.
        device:             Torch device used by the backbone.
    """

    NUM_ACTIONS: int = 2  # {0: coarse, 1: precise}
    OBS_DIM: int = 512    # mean-pooled pre-decoder embedding

    def __init__(
        self,
        sim,
        backbone,
        coarse_steering,
        selection_interval: int = 5,
        max_steps: int = 300,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.sim = sim
        self.backbone = backbone
        self.coarse = coarse_steering
        self.selection_interval = selection_interval
        self.max_steps = max_steps
        self.device = device

        self._step_count: int = 0
        self._robot_state: np.ndarray | None = None
        self._obstacle_states: np.ndarray | None = None
        self._poses: list | None = None
        self._last_action: list | None = None
        self._goal_positions: list | None = None

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, random_obstacles: bool = True) -> np.ndarray:
        """
        Reset the simulation and return the initial observation.

        Returns:
            obs: (512,) mean-pooled pre-decoder embedding (numpy float32).
        """
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, obstacle_states,
        ) = self.sim.reset(random_obstacles=random_obstacles)

        self._step_count = 0
        self._poses = poses
        self._last_action = a
        self._goal_positions = goal_positions
        self._obstacle_states = obstacle_states

        robot_state, _ = self.backbone.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )
        self._robot_state = np.array(robot_state, dtype=np.float32)

        return self._get_obs()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute ``selection_interval`` sim steps with the chosen mode.

        Args:
            action: 0 = coarse steering, 1 = precise (frozen GAT actor).

        Returns:
            obs:    (512,) next observation.
            reward: Accumulated reward over all sub-steps.
            done:   Episode termination flag.
            info:   Dict with diagnostic keys:
                    ``collision``, ``all_reached``, ``timeout``, ``oob``,
                    ``mode``, ``steps_taken``.
        """
        accumulated_reward = 0.0
        mode_cost = MODE_COSTS[action]
        done = False
        info: dict[str, Any] = {
            "collision":   False,
            "all_reached": False,
            "timeout":     False,
            "oob":         False,
            "mode":        action,
            "steps_taken": 0,
        }

        for sub in range(self.selection_interval):
            # ---- Choose sim-input actions based on mode ---------------
            if action == 0:
                # Coarse: least-squares steering — already in sim-input format.
                sim_actions = self.coarse.compute_actions(self._robot_state)
            else:
                # Precise: run frozen GAT actor, convert actor-space → sim-input.
                # Actor outputs raw ∈ [−1, 1]²; simulator expects:
                #   lin_vel  = (raw[0] + 1) / 4   ∈ [0, 0.5]
                #   ang_vel  = raw[1]              ∈ [−1, 1]
                raw_actions, _ = self.backbone.get_embedding_and_actions(
                    self._robot_state, self._obstacle_states
                )
                sim_actions = [
                    [
                        (float(raw_actions[i, 0]) + 1.0) / 4.0,
                        float(raw_actions[i, 1]),
                    ]
                    for i in range(self.sim.num_robots)
                ]

            # ---- Step the simulator ------------------------------------
            (
                poses, distance, cos, sin, collision, goal, a, sim_rewards,
                positions, goal_positions, obstacle_states,
            ) = self.sim.step(sim_actions, None, None)

            self._step_count += 1
            info["steps_taken"] = sub + 1

            # ---- Accumulate reward -------------------------------------
            mean_sim_reward = float(np.mean(sim_rewards))
            accumulated_reward += mean_sim_reward + mode_cost

            # ---- Update cached state ----------------------------------
            robot_state, _ = self.backbone.prepare_state(
                poses, distance, cos, sin, collision, a, goal_positions
            )
            self._robot_state = np.array(robot_state, dtype=np.float32)
            self._obstacle_states = obstacle_states
            self._poses = poses
            self._last_action = a
            self._goal_positions = goal_positions

            # ---- Termination checks -----------------------------------
            if any(collision):
                info["collision"] = True
                done = True
            if all(goal):
                info["all_reached"] = True
                done = True
            if self._step_count >= self.max_steps:
                info["timeout"] = True
                done = True
            if _outside_bounds(poses, self.sim):
                info["oob"] = True
                done = True

            if done:
                break

        return self._get_obs(), accumulated_reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Run GAT forward pass and return mean-pooled pre-decoder embedding.

        Returns:
            (512,) numpy float32 array.
        """
        _, h = self.backbone.get_embedding_and_actions(
            self._robot_state, self._obstacle_states
        )
        # h: (N, 512) CPU tensor — mean over robots → (512,) numpy
        return h.mean(dim=0).numpy().astype(np.float32)
