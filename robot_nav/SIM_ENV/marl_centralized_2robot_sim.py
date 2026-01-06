import random
import numpy as np
import torch
import irsim

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class MARL_2Robot_SIM(SIM_ENV):
    """
    Simplified simulation for testing centralized control with only 2 active robots.
    
    The remaining robots are static and don't receive actions or contribute to learning.
    This allows testing whether centralized control can work in a simpler setting.
    
    Attributes:
        active_robot_ids (list): Indices of the 2 robots being controlled.
        static_robot_ids (list): Indices of robots that remain static.
        coupled_mode (bool): If True, both robots move with same linear velocity.
    """

    def __init__(
        self,
        world_file="multi_robot_world.yaml",
        disable_plotting=False,
        reward_phase=1,
        active_robot_ids=None,
        coupled_mode=False,
    ):
        """
        Initialize the 2-robot simulation environment.

        Args:
            world_file: Path to IRSim world configuration.
            disable_plotting: If True, disables visualization.
            reward_phase: Reward function variant (1 or 2).
            active_robot_ids: List of 2 robot indices to control. Defaults to [0, 1].
            coupled_mode: If True, both robots share linear velocity (coupled control).
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        
        self.num_robots = len(self.env.robot_list)
        self.num_active_robots = 2
        
        # Define which robots are active vs static
        if active_robot_ids is None:
            active_robot_ids = [0, 1]
        assert len(active_robot_ids) == 2, "Must have exactly 2 active robots"
        
        self.active_robot_ids = active_robot_ids
        self.static_robot_ids = [i for i in range(self.num_robots) if i not in active_robot_ids]
        self.coupled_mode = coupled_mode
        
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.x_range = self.env._world.x_range
        self.y_range = self.env._world.y_range
        self.reward_phase = reward_phase

    def step(self, action, connection=None, combined_weights=None):
        """
        Perform a simulation step for active robots only.

        Args:
            action: Actions for 2 active robots [[lin_vel, ang_vel], [lin_vel, ang_vel]].
                   If coupled_mode=True, can also be [lin_vel, ang_vel_1, ang_vel_2].
            connection: Unused, kept for API compatibility.
            combined_weights: Optional attention weights for visualization.

        Returns:
            tuple: (
                active_poses, active_distances, active_coss, active_sins,
                active_collisions, active_goals, action, active_rewards,
                active_positions, active_goal_positions, all_poses, episode_done
            )
        """
        # Build full action list: active robots get actions, static get [0, 0]
        full_action = [[0.0, 0.0] for _ in range(self.num_robots)]
        
        if self.coupled_mode and len(action) == 3:
            # Coupled mode: shared linear velocity, different angular velocities
            lin_vel = action[0]
            full_action[self.active_robot_ids[0]] = [lin_vel, action[1]]
            full_action[self.active_robot_ids[1]] = [lin_vel, action[2]]
        else:
            # Decoupled mode: independent actions for each active robot
            for idx, robot_id in enumerate(self.active_robot_ids):
                full_action[robot_id] = action[idx]
        
        # Step the environment
        self.env.step(action_id=[i for i in range(self.num_robots)], action=full_action)
        self.env.render()

        # Collect data for ALL robots (needed for attention network)
        all_poses = []
        all_robot_states = [
            [self.env.robot_list[i].state[0], self.env.robot_list[i].state[1]]
            for i in range(self.num_robots)
        ]
        
        for i in range(self.num_robots):
            robot_state = self.env.robot_list[i].state
            all_poses.append([
                robot_state[0].item(), 
                robot_state[1].item(), 
                robot_state[2].item()
            ])

        # Collect data for ACTIVE robots only (for rewards/training)
        active_poses = []
        active_distances = []
        active_coss = []
        active_sins = []
        active_collisions = []
        active_goals = []
        active_rewards = []
        active_positions = []
        active_goal_positions = []

        for idx, robot_id in enumerate(self.active_robot_ids):
            robot_state = self.env.robot_list[robot_id].state
            robot_goal = self.env.robot_list[robot_id].goal
            
            # Compute distance to other robots (all robots, not just active)
            closest_robots = [
                np.linalg.norm([
                    all_robot_states[j][0] - robot_state[0],
                    all_robot_states[j][1] - robot_state[1],
                ])
                for j in range(self.num_robots) if j != robot_id
            ]
            
            goal_vector = [
                robot_goal[0].item() - robot_state[0].item(),
                robot_goal[1].item() - robot_state[1].item(),
            ]
            distance = np.linalg.norm(goal_vector)
            goal = self.env.robot_list[robot_id].arrive
            collision = self.env.robot_list[robot_id].collision
            
            pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
            cos, sin = self.cossin(pose_vector, goal_vector)
            
            action_i = full_action[robot_id]
            reward = self.get_reward(
                goal, collision, action_i, closest_robots, distance, self.reward_phase
            )

            position = [robot_state[0].item(), robot_state[1].item()]
            goal_position = [robot_goal[0].item(), robot_goal[1].item()]

            active_poses.append([
                robot_state[0].item(), 
                robot_state[1].item(), 
                robot_state[2].item()
            ])
            active_distances.append(distance)
            active_coss.append(cos)
            active_sins.append(sin)
            active_collisions.append(collision)
            active_goals.append(goal)
            active_rewards.append(reward)
            active_positions.append(position)
            active_goal_positions.append(goal_position)

            # Reset goal if reached
            if goal:
                self.env.robot_list[robot_id].set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                    ],
                )

        # Episode done if any active robot collides or all active goals reached
        episode_done = any(active_collisions) or all(active_goals)
        
        return (
            active_poses,
            active_distances,
            active_coss,
            active_sins,
            active_collisions,
            active_goals,
            action,
            active_rewards,
            active_positions,
            active_goal_positions,
            all_poses,  # Full poses for attention network
            episode_done,
        )

    def reset(self, robot_state=None, robot_goal=None, random_obstacles=False):
        """
        Reset environment: randomize all robot positions, set goals for active robots only.
        
        Returns:
            Same tuple as step(), with initial observations.
        """
        init_states = []
        
        # Randomize positions for ALL robots
        for i, robot in enumerate(self.env.robot_list):
            conflict = True
            while conflict:
                conflict = False
                new_state = [
                    [random.uniform(3, 9)],
                    [random.uniform(3, 9)],
                    [random.uniform(-3.14, 3.14)],
                ]
                pos = [new_state[0][0], new_state[1][0]]
                for loc in init_states:
                    vector = [pos[0] - loc[0], pos[1] - loc[1]]
                    if np.linalg.norm(vector) < 0.8:  # Minimum spacing
                        conflict = True
                        break
            init_states.append(pos)
            robot.set_state(state=np.array(new_state), init=True)
        
        # Set random goals for ACTIVE robots only
        for robot_id in self.active_robot_ids:
            self.env.robot_list[robot_id].set_random_goal(
                obstacle_list=self.env.obstacle_list,
                init=True,
                range_limits=[
                    [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                    [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                ],
            )
        
        self.env.reset()
        
        # Take a zero-action step to get initial observations
        action = [[0.0, 0.0] for _ in range(self.num_active_robots)]
        return self.step(action)

    def get_all_robot_states_for_attention(self):
        """
        Get state information for ALL robots (needed for attention network input).
        
        Returns:
            list: Per-robot state vectors for all N robots.
        """
        states = []
        all_robot_states = [
            [self.env.robot_list[i].state[0], self.env.robot_list[i].state[1]]
            for i in range(self.num_robots)
        ]
        
        for i in range(self.num_robots):
            robot_state = self.env.robot_list[i].state
            robot_goal = self.env.robot_list[i].goal
            
            px, py, theta = robot_state[0].item(), robot_state[1].item(), robot_state[2].item()
            gx, gy = robot_goal[0].item(), robot_goal[1].item()
            
            goal_vector = [gx - px, gy - py]
            distance = np.linalg.norm(goal_vector)
            pose_vector = [np.cos(theta), np.sin(theta)]
            cos, sin = self.cossin(pose_vector, goal_vector)
            
            # For static robots, use zero velocity
            if i in self.static_robot_ids:
                lin_vel, ang_vel = 0.0, 0.0
            else:
                lin_vel, ang_vel = 0.0, 0.0  # Will be updated during training
            
            state = [
                px, py,
                np.cos(theta), np.sin(theta),
                distance / 17,  # Normalized distance
                cos, sin,
                lin_vel, ang_vel,
                gx, gy,
            ]
            states.append(state)
        
        return states

    @staticmethod
    def get_reward(goal, collision, action, closest_robots, distance, phase=1):
        """Compute reward for a single robot."""
        if phase == 1:
            if goal:
                return 100.0
            elif collision:
                return -100.0
            else:
                return -0.1 - distance * 0.01
        else:  # phase 2
            if goal:
                return 100.0
            elif collision:
                return -100.0
            else:
                # Encourage forward motion, penalize proximity
                proximity_penalty = sum([max(0, 1.5 - d) for d in closest_robots]) * 0.5
                return -0.1 - distance * 0.01 - proximity_penalty