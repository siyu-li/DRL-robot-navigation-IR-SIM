"""
MARL Simulation Environment with LiDAR Support.

This module extends the base MARL_SIM to include:
- LiDAR scan extraction per robot
- Enhanced obstacle randomization
- Configurable LiDAR parameters

check how to get lidar reading
check the reward design integrated with obstacles
"""

import irsim
import numpy as np
import random
import torch
import logging
from typing import List, Optional, Tuple, Dict, Any

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class MARL_LIDAR_SIM(SIM_ENV):
    """
    Simulation environment for multi-agent robot navigation with LiDAR support.

    This class extends SIM_ENV to provide a wrapper for multi-robot simulation
    with LiDAR observations, obstacle randomization, and reward computation.

    Attributes:
        env (object): IRSim simulation environment instance.
        robot_goal (np.ndarray): Current goal position(s) for the robots.
        num_robots (int): Number of robots in the environment.
        x_range (tuple): World x-range.
        y_range (tuple): World y-range.
        use_lidar (bool): Whether LiDAR observations are enabled.
        lidar_num_beams (int): Number of LiDAR beams per robot.
        lidar_range_max (float): Maximum LiDAR range.
        random_obstacles (bool): Whether to randomize obstacles on reset.
        num_obstacles (int): Number of obstacles to randomize.
    """

    def __init__(
        self,
        world_file: str = "multi_robot_world_lidar.yaml",
        disable_plotting: bool = False,
        reward_phase: int = 1,
        use_lidar: bool = True,
        lidar_num_beams: int = 180,
        lidar_range_max: float = 7.0,
        random_obstacles: bool = False,
        num_obstacles: int = 5,
        obstacle_size_range: Tuple[float, float] = (0.3, 1.0),
        per_robot_goal_reset: bool = True,

    ):
        """
        Initialize the MARL_LIDAR_SIM environment.

        Args:
            world_file (str): Path to IRSim YAML world configuration.
            disable_plotting (bool): If True, disables all plotting/display.
            reward_phase (int): Reward function variant (1 or 2).
            use_lidar (bool): Whether to include LiDAR in observations.
            lidar_num_beams (int): Expected number of LiDAR beams.
            lidar_range_max (float): Maximum LiDAR range for normalization.
            random_obstacles (bool): Whether to randomize obstacles on reset.
            num_obstacles (int): Number of obstacles to randomize.
            obstacle_size_range (tuple): (min_radius, max_radius) for random obstacles.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.num_robots = len(self.env.robot_list)
        self.x_range = self.env._world.x_range
        self.y_range = self.env._world.y_range
        self.reward_phase = reward_phase

        # LiDAR configuration
        self.use_lidar = use_lidar
        self.lidar_num_beams = lidar_num_beams
        self.lidar_range_max = lidar_range_max

        # Obstacle configuration
        self.random_obstacles_enabled = random_obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_size_range = obstacle_size_range

        # Track obstacle IDs for randomization
        self._obstacle_ids = self._get_obstacle_ids()

        self.per_robot_goal_reset = per_robot_goal_reset

    def _get_obstacle_ids(self) -> List[int]:
        """Get IDs of obstacles that can be randomized."""
        # Obstacles typically start after robots in the entity list
        # Get all obstacle indices
        return [i + self.num_robots for i in range(len(self.env.obstacle_list))]

    def get_lidar_scans(self) -> List[np.ndarray]:
        """
        Get LiDAR scans for all robots.

        Returns:
            List of normalized LiDAR range arrays, one per robot.
            Each array has shape (num_beams,) with values in [0, 1].
        """
        scans = []
        for robot_id in range(self.num_robots):
            scan = self._get_single_lidar_scan(robot_id)
            scans.append(scan)
        return scans

    def _get_single_lidar_scan(self, robot_id: int) -> np.ndarray:
        """
        Get and normalize LiDAR scan for a single robot.

        Args:
            robot_id (int): Robot index.

        Returns:
            Normalized LiDAR ranges of shape (num_beams,) in [0, 1].
        """
        try:
            scan_data = self.env.get_lidar_scan(robot_id)
            return scan_data["ranges"]/self.lidar_range_max

        except Exception as e:
            logging.warning(f"Error getting LiDAR scan for robot {robot_id}: {e}")
            # Return max range (normalized to 1.0) if error
            return np.ones(self.lidar_num_beams, dtype=np.float32)

    def step(
        self,
        action: List[List[float]],
        connection: torch.Tensor,
        combined_weights: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Perform a simulation step for all robots.

        Args:
            action (list): List of [linear_vel, angular_vel] per robot.
            connection (Tensor): Connection logits tensor.
            combined_weights (Tensor, optional): Visualization weights.

        Returns:
            tuple: (poses, distances, coss, sins, collisions, goals, action,
                   rewards, positions, goal_positions, lidar_scans)
                   lidar_scans is included only if use_lidar=True.
        """
        self.env.step(action_id=[i for i in range(self.num_robots)], action=action)
        self.env.render()

        poses = []
        distances = []
        coss = []
        sins = []
        collisions = []
        goals = []
        rewards = []
        positions = []
        goal_positions = []

        robot_states = [
            [self.env.robot_list[i].state[0], self.env.robot_list[i].state[1]]
            for i in range(self.num_robots)
        ]
        lidar_scans = self.get_lidar_scans() if self.use_lidar else None
        for i in range(self.num_robots):
            robot_state = self.env.robot_list[i].state
            closest_robots = [
                np.linalg.norm(
                    [
                        robot_states[j][0] - robot_state[0],
                        robot_states[j][1] - robot_state[1],
                    ]
                )
                for j in range(self.num_robots)
                if j != i
            ]
            robot_goal = self.env.robot_list[i].goal
            goal_vector = [
                robot_goal[0].item() - robot_state[0].item(),
                robot_goal[1].item() - robot_state[1].item(),
            ]
            distance = np.linalg.norm(goal_vector)
            goal = self.env.robot_list[i].arrive
            pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
            cos, sin = self.cossin(pose_vector, goal_vector)
            collision = self.env.robot_list[i].collision
            action_i = action[i]
            lidar_scan = lidar_scans[i] if self.use_lidar else None
            reward = self.get_reward(
                goal, collision, action_i, closest_robots, distance, self.reward_phase, lidar_scan
            )

            position = [robot_state[0].item(), robot_state[1].item()]
            goal_position = [robot_goal[0].item(), robot_goal[1].item()]

            distances.append(distance)
            coss.append(cos)
            sins.append(sin)
            collisions.append(collision)
            goals.append(goal)
            rewards.append(reward)
            positions.append(position)
            poses.append(
                [robot_state[0].item(), robot_state[1].item(), robot_state[2].item()]
            )
            goal_positions.append(goal_position)

            # Visualization of attention weights
            if combined_weights is not None:
                i_weights = combined_weights[i].tolist()
                weight_idx = 0
                for j in range(self.num_robots):
                    if i == j:
                        continue
                    weight = i_weights[weight_idx]
                    weight_idx += 1
                    other_robot_state = self.env.robot_list[j].state
                    other_pos = [
                        other_robot_state[0].item(),
                        other_robot_state[1].item(),
                    ]
                    rx = [position[0], other_pos[0]]
                    ry = [position[1], other_pos[1]]
                    self.env.draw_trajectory(
                        np.array([rx, ry]), refresh=True, linewidth=weight * 2
                    )
        
            
            # Reset goals individually if reached (commented out to allow multiple goals)
            if self.per_robot_goal_reset and goal:
                self.env.robot_list[i].set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                    ],
                )
        
        # Handle all goals reached
        if not self.per_robot_goal_reset and all(goals):
            for i in range(self.num_robots):
                self.env.robot_list[i].set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                    ],
                )

        # Get LiDAR scans if enabled
        if self.use_lidar:
            return (
                poses,
                distances,
                coss,
                sins,
                collisions,
                goals,
                action,
                rewards,
                positions,
                goal_positions,
                lidar_scans,
            )
        else:
            return (
                poses,
                distances,
                coss,
                sins,
                collisions,
                goals,
                action,
                rewards,
                positions,
                goal_positions,
            )

    def reset(
        self,
        robot_state: Optional[List] = None,
        robot_goal: Optional[List] = None,
        random_obstacles: Optional[bool] = None,
        random_obstacle_ids: Optional[List[int]] = None,
    ) -> Tuple:
        """
        Reset the simulation environment.

        Args:
            robot_state (list, optional): Initial robot states.
            robot_goal (list, optional): Initial robot goals.
            random_obstacles (bool, optional): Override random_obstacles setting.
            random_obstacle_ids (list, optional): Specific obstacle IDs to randomize.

        Returns:
            tuple: Initial state including LiDAR scans if use_lidar=True.
        """
        # Use class default if not specified
        if random_obstacles is None:
            random_obstacles = self.random_obstacles_enabled

        # Dynamic world bounds with padding
        x_min = self.x_range[0] + 1
        x_max = self.x_range[1] - 1
        y_min = self.y_range[0] + 1
        y_max = self.y_range[1] - 1

        # Initialize robot positions
        init_states = []
        init_goals = []

        for idx, robot in enumerate(self.env.robot_list):
            conflict = True
            attempts = 0
            while conflict and attempts < 100:
                conflict = False
                robot_state_new = [
                    [random.uniform(x_min, x_max)],
                    [random.uniform(y_min, y_max)],
                    [random.uniform(-3.14, 3.14)],
                ]
                pos = [robot_state_new[0][0], robot_state_new[1][0]]

                # Check distance from other robots
                for loc in init_states:
                    vector = [pos[0] - loc[0], pos[1] - loc[1]]
                    if np.linalg.norm(vector) < 0.6:
                        conflict = True
                        break

                attempts += 1

            init_states.append(pos)
            robot.set_state(state=np.array(robot_state_new), init=True)

        # Randomize obstacles if enabled
        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = self._obstacle_ids

            if len(random_obstacle_ids) > 0:
                self.env.random_obstacle_position(
                    range_low=[x_min, y_min, -3.14],
                    range_high=[x_max, y_max, 3.14],
                    ids=random_obstacle_ids,
                    non_overlapping=True,
                )

        # Set random goals for each robot
        for idx, robot in enumerate(self.env.robot_list):
            goal_conflict = True
            attempts = 0
            while goal_conflict and attempts < 100:
                goal_conflict = False
                robot.set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [x_min, y_min, -3.141592653589793],
                        [x_max, y_max, 3.141592653589793],
                    ],
                )
                goal_pos = [robot.goal[0].item(), robot.goal[1].item()]

                # Check goal doesn't overlap with robot positions
                for loc in init_states:
                    if np.linalg.norm([goal_pos[0] - loc[0], goal_pos[1] - loc[1]]) < 0.5:
                        goal_conflict = True
                        break

                # Check goal doesn't overlap with other goals
                for other_goal in init_goals:
                    if np.linalg.norm([goal_pos[0] - other_goal[0], goal_pos[1] - other_goal[1]]) < 0.5:
                        goal_conflict = True
                        break

                attempts += 1

            init_goals.append([robot.goal[0].item(), robot.goal[1].item()])

        self.env.reset()
        self.robot_goal = self.env.robot.goal

        # Take initial step
        action = [[0.0, 0.0] for _ in range(self.num_robots)]
        con = torch.tensor(
            [[0.0 for _ in range(self.num_robots - 1)] for _ in range(self.num_robots)]
        )

        result = self.step(action, con)

        if self.use_lidar:
            (
                poses,
                distance,
                cos,
                sin,
                _,
                _,
                action,
                reward,
                positions,
                goal_positions,
                lidar_scans,
            ) = result
            return (
                poses,
                distance,
                cos,
                sin,
                [False] * self.num_robots,
                [False] * self.num_robots,
                action,
                reward,
                positions,
                goal_positions,
                lidar_scans,
            )
        else:
            (
                poses,
                distance,
                cos,
                sin,
                _,
                _,
                action,
                reward,
                positions,
                goal_positions,
            ) = result
            return (
                poses,
                distance,
                cos,
                sin,
                [False] * self.num_robots,
                [False] * self.num_robots,
                action,
                reward,
                positions,
                goal_positions,
            )

    def get_reward(
        self,
        goal: bool,
        collision: bool,
        action: List[float],
        closest_robots: List[float],
        distance: float,
        phase: int = 1,
        laser_scan: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute reward for a single robot.

        Args:
            goal (bool): Whether robot reached its goal.
            collision (bool): Whether robot collided.
            action (list): [linear_vel, angular_vel] applied.
            closest_robots (list): Distances to other robots.
            distance (float): Distance to goal.
            phase (int): Reward function variant.

        Returns:
            float: Computed scalar reward.
        """
        def obstacle_penalty(scan, threshold=1.35):
            min_reading = min(scan)
            return (threshold - min_reading) ** 2 if min_reading < threshold else 0.0
        
        match phase:
            case 1:
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 5 / distance
                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add
                    # obstacle proximity penalty
                    obs_pen = 0
                    if laser_scan is not None:
                        obs_pen = obstacle_penalty(laser_scan * self.lidar_range_max, threshold=1.35)
                    return action[0] - 0.5 * abs(action[1]) - cl_pen + r_dist - obs_pen

            case 2:
                if goal:
                    return 70.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (3 - rob) ** 2 if rob < 3 else 0
                        cl_pen += add
                    # obstacle proximity penalty
                    obs_pen = 0
                    if laser_scan is not None:
                        obs_pen = obstacle_penalty(laser_scan * self.lidar_range_max, threshold=1.35)
                    return -0.5 * abs(action[1]) - cl_pen - obs_pen
            case 3: # cap the goal proximity reward
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 10 * np.exp(-distance)
                    # time penalty
                    t_pen = -0.1
                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (5 - rob) ** 2 if rob < 5 else 0
                        cl_pen += add
                    # obstacle proximity penalty
                    obs_pen = 0
                    if laser_scan is not None:
                        obs_pen = obstacle_penalty(laser_scan * self.lidar_range_max, threshold=1.35)
                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_dist + t_pen

            case 4: # cap the goal proximity reward, different robot proximity penalty with case 3
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 10 * np.exp(-distance)
                    # time penalty
                    t_pen = -0.1
                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add
                    # obstacle proximity penalty
                    obs_pen = 0
                    if laser_scan is not None:
                        obs_pen = obstacle_penalty(laser_scan * self.lidar_range_max, threshold=1.35)
                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_dist + t_pen

            case _:
                raise ValueError("Unknown reward phase")

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the environment state.

        Returns:
            dict: Debug info including LiDAR statistics.
        """
        info = {
            "num_robots": self.num_robots,
            "num_obstacles": len(self.env.obstacle_list),
            "use_lidar": self.use_lidar,
            "lidar_num_beams": self.lidar_num_beams,
            "lidar_range_max": self.lidar_range_max,
        }

        if self.use_lidar:
            scans = self.get_lidar_scans()
            info["lidar_stats"] = {
                "min": float(min(s.min() for s in scans)),
                "max": float(max(s.max() for s in scans)),
                "mean": float(np.mean([s.mean() for s in scans])),
            }

        return info
