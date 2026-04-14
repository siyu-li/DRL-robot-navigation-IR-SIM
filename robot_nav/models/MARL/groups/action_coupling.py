"""
Action Coupling for Group-Based Robot Control.

Applies velocity coupling rules to robot groups:
  - All groups (size >= 2): coupled linear velocity = mean or min of scaled
    linear vels (configurable via ``coupling_mode``), individual angular velocity.
  - Size-1 groups: just scale the individual robot's action.
  - Robots NOT in the active group get [0, 0].

Linear velocity scaling: raw in [-1, 1] -> scaled in [0, 0.5] via (v + 1) / 4,
then clamped to a minimum of 0.05.

Two entry points:
  - ``actions_for_group``:          runs the policy forward pass, then couples.
  - ``actions_for_group_from_raw``: uses pre-computed raw actions (no GPU call).
"""

from typing import List

import numpy as np


def actions_for_group_from_raw(
    raw_actions: np.ndarray,
    group: List[int],
    num_robots: int,
    coupling_mode: str = "min",
) -> List[List[float]]:
    """
    Derive coupled actions from pre-computed raw policy output (CPU only).

    This avoids re-running the GPU forward pass for each group.

    Coupling rules:
      - Size 1: individual action, no coupling.
      - Size >= 2: coupled linear velocity (mean or min), individual angular
        velocity.
      - Inactive robots get ``[0.0, 0.0]``.
      - Each scaled linear velocity is clamped to a minimum of ``0.05`` so very
        slow robots cannot stall the whole group.

    Args:
        raw_actions: Pre-computed raw actions, shape ``(num_robots, 2)``,
            values in [-1, 1].
        group: List of robot indices in the active group.
        num_robots: Total number of robots.
        coupling_mode: ``"min"`` (conservative, default) or ``"mean"``.

    Returns:
        Actions for all robots, ``List[List[float]]`` of length ``num_robots``.
    """
    group_size = len(group)
    group_set = set(group)

    # Scaled linear velocities for group members: [-1,1] -> [0,0.5]
    # Clamp to a minimum of 0.05 so very slow robots don't stall the group.
    _vel_threshold = 0.05
    scaled_lin_vels = [max((raw_actions[idx][0] + 1) / 4, _vel_threshold) for idx in group]

    if group_size == 1:
        # Size-1: individual action, no coupling needed
        v = scaled_lin_vels[0]
        robot_idx = group[0]
        a_out = []
        for i in range(num_robots):
            if i == robot_idx:
                a_out.append([v, raw_actions[i][1]])
            else:
                a_out.append([0.0, 0.0])
        return a_out

    # Size >= 2: coupled linear velocity, individual angular velocity.
    if coupling_mode == "min":
        v_coupled = min(scaled_lin_vels)
    else:
        v_coupled = sum(scaled_lin_vels) / len(scaled_lin_vels)

    a_out = []
    for i in range(num_robots):
        if i in group_set:
            a_out.append([v_coupled, raw_actions[i][1]])
        else:
            a_out.append([0.0, 0.0])
    return a_out


def actions_for_group(
    policy,
    robot_obs: np.ndarray,
    obstacle_obs: np.ndarray,
    group: List[int],
    num_robots: int,
    add_noise: bool = False,
    coupling_mode: str = "min",
) -> List[List[float]]:
    """
    Run the decentralized policy forward pass and apply group action coupling.

    Convenience wrapper: calls ``policy.get_action`` then delegates to
    ``actions_for_group_from_raw``.

    Args:
        policy: TD3Obstacle (or compatible) policy with
            ``get_action(robot_obs, obstacle_obs, add_noise) -> (actions, ...)``.
        robot_obs: Robot observations, shape ``(num_robots, state_dim)``.
        obstacle_obs: Obstacle observations, shape ``(num_obstacles, obs_dim)``.
        group: List of robot indices in the active group.
        num_robots: Total number of robots.
        add_noise: Whether to add exploration noise to the policy output.
        coupling_mode: ``"min"`` (conservative, default) or ``"mean"``.

    Returns:
        Actions for all robots, ``List[List[float]]`` of length ``num_robots``.
    """
    action, _ = policy.get_action(robot_obs, obstacle_obs, add_noise=add_noise)
    return actions_for_group_from_raw(
        raw_actions=action,
        group=group,
        num_robots=num_robots,
        coupling_mode=coupling_mode,
    )
