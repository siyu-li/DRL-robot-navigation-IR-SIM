"""
Action Coupling for Group-Based Robot Control.

Applies velocity coupling rules to robot groups:
  - All groups (size ≥ 2): coupled linear velocity = mean of scaled linear vels.
  - Groups with size > rotation_coupling_threshold: also couple angular velocity
    (average of group members).
  - Size-1 groups: just scale the individual robot's action.
  - Robots NOT in the active group get [0, 0].

Linear velocity scaling: raw ∈ [-1, 1] → scaled ∈ [0, 0.5] via (v + 1) / 4.

Using mean (instead of min) allows groups containing both reached and unreached
robots to still move, since a reached robot's near-zero velocity no longer
drags the entire group to a halt.

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
    use_rotation_coupling: bool = True,
    rotation_coupling_threshold: int = 3,
) -> List[List[float]]:
    """
    Derive coupled actions from pre-computed raw policy output (CPU only).

    This avoids re-running the GPU forward pass for each group.

    Coupling rules:
      - Size 1: individual action, no coupling.
      - Size 2–3: coupled linear velocity (mean), individual angular velocity.
      - Size > threshold (if ``use_rotation_coupling``): coupled linear (mean)
        AND coupled angular (mean).
      - Inactive robots get ``[0.0, 0.0]``.

    Args:
        raw_actions: Pre-computed raw actions, shape ``(num_robots, 2)``,
            values in [-1, 1].
        group: List of robot indices in the active group.
        num_robots: Total number of robots.
        use_rotation_coupling: Whether to couple angular velocity for large groups.
        rotation_coupling_threshold: Groups with ``size > threshold`` use
            coupled angular velocity.  Default: 3.

    Returns:
        Actions for all robots, ``List[List[float]]`` of length ``num_robots``.
    """
    group_size = len(group)
    group_set = set(group)

    # Scaled linear velocities for group members: [-1,1] -> [0,0.5]
    scaled_lin_vels = [(raw_actions[idx][0] + 1) / 4 for idx in group]

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

    # Size >= 2: coupled linear velocity = mean
    v_coupled = sum(scaled_lin_vels) / len(scaled_lin_vels)

    # Angular velocity coupling for large groups
    if group_size > rotation_coupling_threshold and use_rotation_coupling:
        ang_vels = [raw_actions[idx][1] for idx in group]
        w_coupled = sum(ang_vels) / len(ang_vels)
    else:
        w_coupled = None  # per-robot angular velocity

    a_out = []
    for i in range(num_robots):
        if i in group_set:
            w = w_coupled if w_coupled is not None else raw_actions[i][1]
            a_out.append([v_coupled, w])
        else:
            a_out.append([0.0, 0.0])
    return a_out


def actions_for_group(
    policy,
    robot_obs: np.ndarray,
    obstacle_obs: np.ndarray,
    group: List[int],
    num_robots: int,
    use_rotation_coupling: bool = True,
    rotation_coupling_threshold: int = 3,
    add_noise: bool = False,
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
        use_rotation_coupling: Whether to couple angular velocity for large groups.
        rotation_coupling_threshold: Groups with ``size > threshold`` use
            coupled angular velocity.  Default: 3.
        add_noise: Whether to add exploration noise to the policy output.

    Returns:
        Actions for all robots, ``List[List[float]]`` of length ``num_robots``.
    """
    action, _ = policy.get_action(robot_obs, obstacle_obs, add_noise=add_noise)
    return actions_for_group_from_raw(
        raw_actions=action,
        group=group,
        num_robots=num_robots,
        use_rotation_coupling=use_rotation_coupling,
        rotation_coupling_threshold=rotation_coupling_threshold,
    )
