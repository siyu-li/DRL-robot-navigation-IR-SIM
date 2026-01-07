from typing import List
from tqdm import tqdm
import yaml
import torch

from robot_nav.models.RCPG.RCPG import RCPG
from robot_nav.replay_buffer import ReplayBuffer, RolloutReplayBuffer
from robot_nav.models.PPO.PPO import PPO


class Pretraining:
    """
    Handles loading of offline experience data and pretraining of a reinforcement learning model.

    Attributes:
        file_names (List[str]): List of YAML files containing pre-recorded environment samples.
        model (object): The model with `prepare_state` and `train` methods.
        replay_buffer (object): The buffer used to store experiences for training.
        reward_function (callable): Function to compute the reward from the environment state.
    """

    def __init__(
        self,
        file_names: List[str],
        model: object,
        replay_buffer: object,
        reward_function,
    ):
        self.file_names = file_names
        self.model = model
        self.replay_buffer = replay_buffer
        self.reward_function = reward_function

    def load_buffer_centralized(self, num_robots=5, reward_phase=1):
        """
        Load samples from MARL data files and populate the replay buffer for centralized training.

        Expected YAML format per sample (saved after each sim.step):
            poses: [[x, y, theta], ...] - state AFTER taking actions
            distances: [d1, d2, ...] per robot
            cos: [c1, c2, ...] per robot  
            sin: [s1, s2, ...] per robot
            collisions: [bool, ...] per robot
            goals: [bool, ...] per robot
            actions: [[lin_vel, ang_vel], ...] - action that LED TO this state
            goal_positions: [[gx, gy], ...] per robot

        Data interpretation:
            sample[i].poses = state after taking sample[i].actions
            To form (state, action, next_state):
                state = sample[i].poses
                action = sample[i+1].actions  (action taken FROM sample[i] to reach sample[i+1])
                next_state = sample[i+1].poses

        Terminal condition: ANY robot collision = episode terminal.
        Reward: Per-robot rewards based on resulting state.

        Args:
            num_robots (int): Number of robots in the data. Defaults to 5.
            reward_phase (int): Reward function phase (1 or 2). Defaults to 1.

        Returns:
            (object): The populated replay buffer.
        """
        import numpy as np

        for file_name in self.file_names:
            print("Loading file: ", file_name)
            with open(file_name, "r") as file:
                samples = yaml.full_load(file)
                
                for i in tqdm(range(1, len(samples) - 1)):
                    sample = samples[i]
                    next_sample = samples[i + 1]
                    
                    # Current state (from sample[i])
                    poses = sample["poses"]
                    distances = sample["distances"]
                    cos_vals = sample["cos"]
                    sin_vals = sample["sin"]
                    collisions = sample["collisions"]
                    goals = sample["goals"]
                    goal_positions = sample["goal_positions"]
                    
                    # Action that transitions from sample[i] to sample[i+1]
                    # This is stored in sample[i+1] as the action that led to that state
                    actions = next_sample["actions"]

                    # Terminal if ANY robot collided in current state
                    terminal = any(collisions)
                    
                    if terminal:
                        continue

                    # Prepare current state using model's prepare_state
                    # Note: prepare_state needs an action for the state vector (last action taken)
                    # Use sample[i].actions since that's what led to sample[i].poses
                    # Convert raw actions [-1,1] to a_in format: [(a[0]+1)/4, a[1]]
                    current_actions_raw = sample["actions"]
                    current_actions_converted = [
                        [(a[0] + 1) / 4, a[1]] for a in current_actions_raw
                    ]
                    state = self.model.prepare_state(
                        poses, distances, cos_vals, sin_vals, collisions, 
                        current_actions_converted, goal_positions
                    )

                    # Next state data (from sample[i+1])
                    next_poses = next_sample["poses"]
                    next_distances = next_sample["distances"]
                    next_cos = next_sample["cos"]
                    next_sin = next_sample["sin"]
                    next_collisions = next_sample["collisions"]
                    next_goals = next_sample["goals"]
                    next_goal_positions = next_sample["goal_positions"]

                    # Terminal if ANY robot collided in next state
                    next_terminal = any(next_collisions)

                    # Convert actions (raw [-1,1]) to a_in format: [(a[0]+1)/4, a[1]]
                    actions_converted = [
                        [(a[0] + 1) / 4, a[1]] for a in actions
                    ]

                    # Prepare next state
                    next_state = self.model.prepare_state(
                        next_poses, next_distances, next_cos, next_sin,
                        next_collisions, actions_converted, next_goal_positions
                    )

                    # Calculate per-robot rewards based on next state
                    rewards = []
                    robot_positions = [[p[0], p[1]] for p in next_poses]
                    
                    for r in range(num_robots):
                        # Calculate distances to other robots
                        closest_robots = []
                        for other_r in range(num_robots):
                            if other_r != r:
                                dist = np.linalg.norm([
                                    robot_positions[other_r][0] - robot_positions[r][0],
                                    robot_positions[other_r][1] - robot_positions[r][1]
                                ])
                                closest_robots.append(dist)
                        
                        reward = self.reward_function(
                            next_goals[r],
                            next_collisions[r],
                            actions[r],  # action that caused this transition
                            closest_robots,
                            next_distances[r],
                            reward_phase,
                        )
                        rewards.append(reward)

                    # Create terminal flag list (same value for all robots, as centralized expects)
                    terminal_flags = [next_terminal] * num_robots

                    self.replay_buffer.add(
                        state, actions, rewards, terminal_flags, next_state
                    )

        return self.replay_buffer

    def load_buffer(self):
        """
        Load samples from the specified files and populate the replay buffer.

        Returns:
            (object): The populated replay buffer.
        """
        for file_name in self.file_names:
            print("Loading file: ", file_name)
            with open(file_name, "r") as file:
                samples = yaml.full_load(file)
                for i in tqdm(range(1, len(samples) - 1)):
                    sample = samples[i]
                    latest_scan = sample["latest_scan"]
                    distance = sample["distance"]
                    cos = sample["cos"]
                    sin = sample["sin"]
                    collision = sample["collision"]
                    goal = sample["goal"]
                    action = sample["action"]

                    state, terminal = self.model.prepare_state(
                        latest_scan, distance, cos, sin, collision, goal, action
                    )

                    if terminal:
                        continue

                    next_sample = samples[i + 1]
                    next_latest_scan = next_sample["latest_scan"]
                    next_distance = next_sample["distance"]
                    next_cos = next_sample["cos"]
                    next_sin = next_sample["sin"]
                    next_collision = next_sample["collision"]
                    next_goal = next_sample["goal"]
                    next_action = next_sample["action"]
                    next_state, next_terminal = self.model.prepare_state(
                        next_latest_scan,
                        next_distance,
                        next_cos,
                        next_sin,
                        next_collision,
                        next_goal,
                        next_action,
                    )
                    reward = self.reward_function(
                        next_goal, next_collision, action, next_latest_scan
                    )
                    self.replay_buffer.add(
                        state, action, reward, next_terminal, next_state
                    )

        return self.replay_buffer

    def train(
        self,
        pretraining_iterations,
        replay_buffer,
        iterations,
        batch_size,
    ):
        """
        Run pretraining on the model using the replay buffer.

        Args:
            pretraining_iterations (int): Number of outer loop iterations for pretraining.
            replay_buffer (object): Buffer to sample training batches from.
            iterations (int): Number of training steps per pretraining iteration.
            batch_size (int): Batch size used during training.
        """
        print("Running Pretraining")
        for _ in tqdm(range(pretraining_iterations)):
            self.model.train(
                replay_buffer=replay_buffer,
                iterations=iterations,
                batch_size=batch_size,
            )
        print("Model Pretrained")


def get_buffer(
    model,
    sim,
    load_saved_buffer,
    pretrain,
    pretraining_iterations,
    training_iterations,
    batch_size,
    buffer_size=50000,
    random_seed=666,
    file_names=["robot_nav/assets/data.yml"],
    history_len=10,
):
    """
    Get or construct the replay buffer depending on model type and training configuration.

    Args:
        model (object): The RL model, can be PPO, RCPG, or other.
        sim (object): Simulation environment with a `get_reward` function.
        load_saved_buffer (bool): Whether to load experiences from file.
        pretrain (bool): Whether to run pretraining using the buffer.
        pretraining_iterations (int): Number of outer loop iterations for pretraining.
        training_iterations (int): Number of iterations in each training loop.
        batch_size (int): Size of the training batch.
        buffer_size (int, optional): Maximum size of the buffer. Defaults to 50000.
        random_seed (int, optional): Seed for reproducibility. Defaults to 666.
        file_names (List[str], optional): List of YAML data file paths. Defaults to ["robot_nav/assets/data.yml"].
        history_len (int, optional): Used for RCPG buffer configuration. Defaults to 10.

    Returns:
        (object): The initialized and optionally pre-populated replay buffer.
    """
    if isinstance(model, PPO):
        return model.buffer

    if isinstance(model, RCPG):
        replay_buffer = RolloutReplayBuffer(
            buffer_size=buffer_size, random_seed=random_seed, history_len=history_len
        )
    else:
        replay_buffer = ReplayBuffer(buffer_size=buffer_size, random_seed=random_seed)

    if pretrain:
        assert (
            load_saved_buffer
        ), "To pre-train model, load_saved_buffer must be set to True"

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=file_names,
            model=model,
            replay_buffer=replay_buffer,
            reward_function=sim.get_reward,
        )  # instantiate pre-trainind
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training

    return replay_buffer


def get_max_bound(
    next_state,
    discount,
    max_ang_vel,
    max_lin_vel,
    time_step,
    distance_norm,
    goal_reward,
    reward,
    done,
    device,
):
    """
    Estimate the maximum possible return (upper bound) from the next state onward.

    This is used in constrained RL or safe policy optimization where a conservative
    estimate of return is useful for policy updates.

    Args:
        next_state (torch.Tensor): Tensor of next state observations.
        discount (float): Discount factor for future rewards.
        max_ang_vel (float): Maximum angular velocity of the agent.
        max_lin_vel (float): Maximum linear velocity of the agent.
        time_step (float): Duration of one time step.
        distance_norm (float): Normalization factor for distance.
        goal_reward (float): Reward received upon reaching the goal.
        reward (torch.Tensor): Immediate reward from the environment.
        done (torch.Tensor): Binary tensor indicating episode termination.
        device (torch.device): PyTorch device for computation.

    Returns:
        (torch.Tensor): Maximum return bound for each sample in the batch.
    """
    next_state = next_state.clone()  # Prevents in-place modifications
    reward = reward.clone()  # Ensures original reward is unchanged
    done = done.clone()
    cos = next_state[:, -4]
    sin = next_state[:, -3]
    theta = torch.atan2(sin, cos)

    # Compute turning steps
    turn_steps = theta.abs() / (max_ang_vel * time_step)
    full_turn_steps = torch.floor(turn_steps)
    turn_rew = -max_ang_vel * discount**full_turn_steps
    turn_rew[full_turn_steps == 0] = 0  # Handle zero case
    final_turn_rew = -(discount ** (full_turn_steps + 1)) * (
        turn_steps - full_turn_steps
    )
    full_turn_rew = turn_rew + final_turn_rew

    # Compute distance-based steps
    full_turn_steps += 1  # Account for the final turn step
    distances = (next_state[:, -5] * distance_norm) / (max_lin_vel * time_step)
    final_steps = torch.ceil(distances) + full_turn_steps
    inter_steps = torch.trunc(distances) + full_turn_steps

    final_rew = goal_reward * discount**final_steps

    # Compute intermediate rewards using a sum of discounted steps
    max_inter_steps = inter_steps.max().int().item()
    discount_exponents = discount ** torch.arange(1, max_inter_steps + 1, device=device)
    inter_rew = torch.stack(
        [
            (max_lin_vel * discount_exponents[int(start) + 1 : int(steps)]).sum()
            for start, steps in zip(full_turn_steps, inter_steps)
        ]
    )
    # Compute final max bound
    max_bound = reward + (1 - done) * (full_turn_rew + final_rew + inter_rew).view(
        -1, 1
    )
    return max_bound


class MARLDataSaver:
    """
    Saves MARL experience data to YAML files for later use in pretraining.
    
    The data format is compatible with Pretraining.load_buffer_centralized().
    """
    
    def __init__(self, filepath="robot_nav/assets/marl_data.yml"):
        """
        Initialize the MARL data saver.
        
        Args:
            filepath (str): Path to save the YAML file.
        """
        self.filepath = filepath
        self.samples = {}
        self.sample_count = 0
    
    def _to_list(self, data):
        """Convert numpy arrays to nested lists for YAML serialization."""
        import numpy as np
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [self._to_list(item) for item in data]
        elif isinstance(data, (np.floating, np.integer)):
            return float(data)
        elif isinstance(data, (np.bool_, bool)):
            return bool(data)
        else:
            return data
    
    def add(self, poses, distances, cos_vals, sin_vals, collisions, goals, actions, goal_positions):
        """
        Add a sample to the data collection.
        
        Args:
            poses: List of [x, y, theta] for each robot.
            distances: List of distances to goal for each robot.
            cos_vals: List of cos(heading error) for each robot.
            sin_vals: List of sin(heading error) for each robot.
            collisions: List of collision flags for each robot.
            goals: List of goal reached flags for each robot.
            actions: List of [lin_vel, ang_vel] for each robot (raw model output, [-1, 1] range).
            goal_positions: List of [gx, gy] for each robot.
        """
        self.sample_count += 1
        self.samples[self.sample_count] = {
            "poses": self._to_list(poses),
            "distances": self._to_list(distances),
            "cos": self._to_list(cos_vals),
            "sin": self._to_list(sin_vals),
            "collisions": self._to_list(collisions),
            "goals": self._to_list(goals),
            "actions": self._to_list(actions),
            "goal_positions": self._to_list(goal_positions),
        }
    
    def save(self):
        """Save all collected samples to the YAML file."""
        with open(self.filepath, "w") as f:
            yaml.dump(self.samples, f)
        print(f"Saved {self.sample_count} samples to {self.filepath}")
    
    def clear(self):
        """Clear all collected samples."""
        self.samples = {}
        self.sample_count = 0
