"""
MARL Simulation Environment with Obstacle Graph Nodes.

This module extends MARL_SIM to include obstacle information for the
graph attention network. Uses Shapely geometry.distance() for efficient
clearance computation.

Key features:
- Extracts obstacle positions and headings from IR-SIM
- Computes robot-obstacle clearance using Shapely geometry
- Provides obstacle state observations for the GAT
"""

import warnings
import irsim
import numpy as np
import random
import torch
import logging
from shapely.geometry import Point
from typing import List, Tuple, Optional

from robot_nav.SIM_ENV.sim_env import SIM_ENV

# Suppress matplotlib color redundancy warning from irsim
warnings.filterwarnings("ignore", message="color is redundantly defined")


class MARL_SIM_OBSTACLE(SIM_ENV):
    """
    Simulation environment for multi-agent robot navigation with obstacle graph nodes.

    Extends SIM_ENV to provide obstacle observations for the graph attention network.
    Obstacles are represented as static nodes with position and heading information.

    Attributes:
        env (object): IRSim simulation environment instance.
        robot_goal (np.ndarray): Current goal position(s) for the robots.
        num_robots (int): Number of robots in the environment.
        num_obstacles (int): Number of obstacles in the environment.
        x_range (tuple): World x-range.
        y_range (tuple): World y-range.
    """

    def __init__(
        self,
        world_file: str = "multi_robot_world_lidar.yaml",
        disable_plotting: bool = False,
        reward_phase: int = 1,
        per_robot_goal_reset: bool = True,
        obstacle_proximity_threshold: float = 1.5,
    ):
        """
        Initialize the MARL_SIM_OBSTACLE environment.

        Args:
            world_file (str): Path to an IRSim YAML world configuration.
            disable_plotting (bool): If True, disables all IRSim plotting.
            reward_phase (int): Selects the reward function variant (1, 2, or 3).
            per_robot_goal_reset (bool): If True, reset individual robot goals when reached.
            obstacle_proximity_threshold (float): Distance threshold for obstacle penalty in reward.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.num_robots = len(self.env.robot_list)
        self.num_obstacles = len(self.env.obstacle_list)
        self.x_range = self.env._world.x_range
        self.y_range = self.env._world.y_range
        self.reward_phase = reward_phase
        self.per_robot_goal_reset = per_robot_goal_reset
        self.obstacle_proximity_threshold = obstacle_proximity_threshold
        
        # Track previous distances for progress-based reward
        self.prev_distances = [None] * self.num_robots

    def get_obstacle_states(self) -> np.ndarray:
        """
        Get obstacle state observations for graph nodes.

        Returns:
            np.ndarray: Obstacle states of shape (num_obstacles, 4).
                Each row: [x, y, cos(heading), sin(heading)]
                For static obstacles, heading is derived from their orientation.
        """
        obstacle_states = []
        for obs in self.env.obstacle_list:
            # Position (centroid)
            pos = obs.position.flatten()
            ox, oy = pos[0], pos[1]

            # Heading (from state if available, else default to 0)
            if hasattr(obs, 'state') and len(obs.state) > 2:
                theta = obs.state[2, 0]
            else:
                theta = 0.0

            cos_h = np.cos(theta)
            sin_h = np.sin(theta)

            obstacle_states.append([ox, oy, cos_h, sin_h])

        return np.array(obstacle_states) if obstacle_states else np.zeros((0, 4))

    def get_robot_obstacle_clearances(self) -> List[List[float]]:
        """
        Compute clearance (distance to boundary) from each robot to each obstacle.

        Uses Shapely's geometry.distance() for efficient computation.
        This gives the distance from robot center to the obstacle boundary.

        Returns:
            List[List[float]]: Clearances of shape (num_robots, num_obstacles).
                clearances[i][j] = distance from robot i center to obstacle j boundary.
        """
        clearances = []
        for robot in self.env.robot_list:
            robot_pos = robot.position.flatten()
            robot_point = Point(robot_pos[0], robot_pos[1])

            robot_clearances = []
            for obs in self.env.obstacle_list:
                # Shapely distance() returns distance from point to polygon boundary
                clearance = obs.geometry.distance(robot_point)
                robot_clearances.append(clearance)

            clearances.append(robot_clearances)

        return clearances

    def get_min_obstacle_clearance(self, robot_idx: int) -> float:
        """
        Get minimum clearance from a robot to any obstacle.

        Args:
            robot_idx (int): Index of the robot.

        Returns:
            float: Minimum clearance to any obstacle (or inf if no obstacles).
        """
        if self.num_obstacles == 0:
            return float('inf')

        robot = self.env.robot_list[robot_idx]
        robot_pos = robot.position.flatten()
        robot_point = Point(robot_pos[0], robot_pos[1])

        min_clearance = float('inf')
        for obs in self.env.obstacle_list:
            clearance = obs.geometry.distance(robot_point)
            min_clearance = min(min_clearance, clearance)

        return min_clearance

    def step(self, action, connection=None, combined_weights=None):
        """
        Perform a simulation step for all robots.

        Args:
            action (list): Actions for each robot [[lin_vel, ang_vel], ...].
            connection (Tensor or None): Connection logits (unused, for compatibility).
            combined_weights (Tensor or None): Visualization weights.

        Returns:
            tuple: (
                poses, distances, coss, sins, collisions, goals,
                action, rewards, positions, goal_positions, obstacle_states
            )
            obstacle_states (np.ndarray): Shape (num_obstacles, 4).
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

        # Pre-compute clearances for reward computation
        clearances = self.get_robot_obstacle_clearances()

        for i in range(self.num_robots):
            robot_state = self.env.robot_list[i].state
            robot_goal = self.env.robot_list[i].goal

            # Distances to other robots
            closest_robots = [
                np.linalg.norm([
                    robot_states[j][0] - robot_state[0],
                    robot_states[j][1] - robot_state[1],
                ])
                for j in range(self.num_robots)
                if j != i
            ]

            # Distance to goal
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

            # Min clearance to obstacles
            min_clearance = min(clearances[i]) if clearances[i] else float('inf')

            # Get previous distance for progress-based reward
            prev_distance = self.prev_distances[i]

            # Compute reward with obstacle penalty
            reward = self.get_reward(
                goal, collision, action_i, closest_robots, distance,
                min_clearance, self.obstacle_proximity_threshold, self.reward_phase,
                prev_distance=prev_distance
            )
            
            # Update previous distance for next step
            self.prev_distances[i] = distance

            position = [robot_state[0].item(), robot_state[1].item()]
            goal_position = [robot_goal[0].item(), robot_goal[1].item()]

            distances.append(distance)
            coss.append(cos)
            sins.append(sin)
            collisions.append(collision)
            goals.append(goal)
            rewards.append(reward)
            positions.append(position)
            poses.append([
                robot_state[0].item(),
                robot_state[1].item(),
                robot_state[2].item()
            ])
            goal_positions.append(goal_position)

            # Visualization (if weights provided)
            if combined_weights is not None:
                # combined_weights has shape (batch, n_robots, n_total) where n_total = n_robots + n_obstacles
                # Squeeze batch dimension if present
                if combined_weights.dim() == 3:
                    weights = combined_weights[0]  # Remove batch dim -> (n_robots, n_total)
                else:
                    weights = combined_weights  # Already (n_robots, n_total)
                
                i_weights_all = weights[i].tolist()
                n_weights = len(i_weights_all)
                
                # Visualize robot-robot attention weights (blue lines)
                for j in range(self.num_robots):
                    if i == j:
                        continue
                    if j >= n_weights:
                        break
                    weight = i_weights_all[j]  # Direct indexing by j
                    
                    # Ensure weight is a scalar float
                    if isinstance(weight, (list, np.ndarray)):
                        weight = float(weight[0] if len(weight) > 0 else 0.0)
                    else:
                        weight = float(weight)
                    
                    if weight > 0.01:  # Only draw if weight is significant
                        other_robot_state = self.env.robot_list[j].state
                        other_pos = [other_robot_state[0].item(), other_robot_state[1].item()]
                        rx = [position[0], other_pos[0]]
                        ry = [position[1], other_pos[1]]
                        self.env.draw_trajectory(
                            np.array([rx, ry]), refresh=True, linewidth=weight * 2, color='blue', alpha=0.6
                        )
                
                # Visualize robot-obstacle attention weights (red lines)
                # Number of obstacle weights available in combined_weights
                n_obs_weights = n_weights - self.num_robots
                for k in range(min(self.num_obstacles, n_obs_weights)):
                    obs_idx = self.num_robots + k
                    weight = i_weights_all[obs_idx]
                    
                    # Ensure weight is a scalar float
                    if isinstance(weight, (list, np.ndarray)):
                        weight = float(weight[0] if len(weight) > 0 else 0.0)
                    else:
                        weight = float(weight)
                    
                    if weight > 0.01:  # Only draw if weight is significant
                        obs_pos = self.env.obstacle_list[k].position.flatten()
                        obs_x, obs_y = obs_pos[0], obs_pos[1]
                        rx = [position[0], obs_x]
                        ry = [position[1], obs_y]
                        self.env.draw_trajectory(
                            np.array([rx, ry]), refresh=True, linewidth=weight * 2, color='red', alpha=0.6
                        )

            # Reset goal if reached
            if self.per_robot_goal_reset and goal:
                self.env.robot_list[i].set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -np.pi],
                        [self.x_range[1] - 1, self.y_range[1] - 1, np.pi],
                    ],
                )
                # Reset prev_distance for this robot to avoid spurious progress reward
                # when goal changes (new goal has different distance)
                self.prev_distances[i] = None

        # Get obstacle states
        obstacle_states = self.get_obstacle_states()

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
            obstacle_states,
        )

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=False,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment.

        Args:
            robot_state: Initial robot state(s). If None, random states are assigned.
            robot_goal: Fixed goal position(s). If None, random goals are generated.
            random_obstacles (bool): If True, reposition obstacles randomly.
            random_obstacle_ids (list or None): IDs of obstacles to randomize.

        Returns:
            tuple: (
                poses, distances, coss, sins, collisions, goals,
                action, rewards, positions, goal_positions, obstacle_states
            )
        """
        x_min = self.x_range[0] + 1
        x_max = self.x_range[1] - 1
        y_min = self.y_range[0] + 1
        y_max = self.y_range[1] - 1

        init_states = []
        for robot in self.env.robot_list:
            conflict = True
            while conflict:
                conflict = False
                robot_state = [
                    [random.uniform(x_min, x_max)],
                    [random.uniform(y_min, y_max)],
                    [random.uniform(-np.pi, np.pi)],
                ]
                pos = [robot_state[0][0], robot_state[1][0]]

                # Check robot-robot spacing
                for loc in init_states:
                    vector = [pos[0] - loc[0], pos[1] - loc[1]]
                    if np.linalg.norm(vector) < 0.6:
                        conflict = True
                        break

                # Check robot-obstacle clearance
                if not conflict:
                    robot_point = Point(pos[0], pos[1])
                    for obs in self.env.obstacle_list:
                        if obs.geometry.distance(robot_point) < 0.5:
                            conflict = True
                            break

            init_states.append(pos)
            robot.set_state(state=np.array(robot_state), init=True)

        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + self.num_robots for i in range(7)]
            self.env.random_obstacle_position(
                range_low=[self.x_range[0], self.y_range[0], -np.pi],
                range_high=[self.x_range[1], self.y_range[1], np.pi],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )

        for robot in self.env.robot_list:
            if robot_goal is None:
                robot.set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -np.pi],
                        [self.x_range[1] - 1, self.y_range[1] - 1, np.pi],
                    ],
                )
            else:
                robot.set_goal(np.array(robot_goal), init=True)

        self.env.reset()
        self.robot_goal = self.env.robot.goal
        
        # Reset previous distances for progress-based reward
        self.prev_distances = [None] * self.num_robots

        action = [[0.0, 0.0] for _ in range(self.num_robots)]
        (
            poses, distances, coss, sins, _, _, action, rewards,
            positions, goal_positions, obstacle_states
        ) = self.step(action, None)

        return (
            poses,
            distances,
            coss,
            sins,
            [False] * self.num_robots,
            [False] * self.num_robots,
            action,
            rewards,
            positions,
            goal_positions,
            obstacle_states,
        )

    @staticmethod
    def get_reward(
        goal, collision, action, closest_robots, distance,
        min_obstacle_clearance, obstacle_threshold, phase=3,
        prev_distance=None
    ):
        """
        Compute the reward for a single robot with obstacle penalty.

        Args:
            goal (bool): Whether the robot reached its goal.
            collision (bool): Whether the robot collided.
            action (list): [linear_velocity, angular_velocity].
            closest_robots (list): Distances to other robots.
            distance (float): Current distance to the goal.
            min_obstacle_clearance (float): Minimum clearance to any obstacle.
            obstacle_threshold (float): Distance threshold for obstacle penalty.
            phase (int): Reward configuration (1, 2, 3, or 5).
            prev_distance (float or None): Previous distance to goal (for progress reward).

        Returns:
            float: Computed scalar reward.
        """
        match phase:
            case 1:
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 1.5 / distance

                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    # Obstacle proximity penalty
                    obs_pen = 0
                    if min_obstacle_clearance < obstacle_threshold:
                        obs_pen = (obstacle_threshold - min_obstacle_clearance) ** 2

                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_dist

            case 2:
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 
                else:
                    return -0.1

            case 3:
                # Phase 3: Stronger obstacle avoidance emphasis
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 10 * np.exp(-distance)

                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    # Stronger obstacle penalty with exponential scaling
                    obs_pen = 0
                    if min_obstacle_clearance < obstacle_threshold:
                        obs_pen = 2.0 * (obstacle_threshold - min_obstacle_clearance) ** 2

                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_dist
            
            case 5:
                # Phase 5: Progress-based reward (change in distance to goal)
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    # Progress reward: positive if moving closer, negative if moving away
                    k_p = 5.0  # Progress reward scaling factor
                    if prev_distance is not None:
                        progress = prev_distance - distance
                        r_progress = k_p * progress
                    else:
                        # First step: no previous distance, use small baseline
                        r_progress = 0.0

                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    # Obstacle proximity penalty
                    obs_pen = 0
                    if min_obstacle_clearance < obstacle_threshold:
                        obs_pen = 2.0 * (obstacle_threshold - min_obstacle_clearance) ** 2

                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_progress
                
            case _:
                raise ValueError(f"Unknown reward phase: {phase}")
