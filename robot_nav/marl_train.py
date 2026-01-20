from pathlib import Path

from robot_nav.models.MARL.marlTD3.marlTD3 import TD3

import torch
import numpy as np
import logging
from robot_nav.SIM_ENV.marl_sim import MARL_SIM
from robot_nav.utils import get_buffer
from robot_nav.utils import MARLDataSaver
# Suppress IRSim warnings
logging.getLogger('irsim').setLevel(logging.ERROR)


def outside_of_bounds(poses, sim):
    """
    Check if any robot is outside the defined world boundaries.

    Args:
        poses (list): List of [x, y, theta] poses for each robot.

    Returns:
        bool: True if any robot is outside world boundaries.
    """
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def main(args=None):
    """Main training function"""

    # ---- Hyperparameters and setup ----
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 11  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    max_epochs = 2000  # max number of epochs
    epoch = 1  # starting epoch number
    episode = 0  # starting episode number
    train_every_n = 10  # train and update network parameters every n episodes
    training_iterations = 80  # how many batches to use for single training cycle
    batch_size = 16  # batch size for each training iteration
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    load_saved_buffer = False  # whether to load experiences from assets/data.yml
    pretrain = False  # whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
    pretraining_iterations = (
        10  # number of training iterations to run during pre-training
    )
    save_every = 5  # save the model every n training cycles
    save_data = False  # whether to save experience data to YAML for centralized pretraining
    data_save_path = "robot_nav/assets/marl_data.yml"  # path to save experience data

    # ---- Environment Hyperparameters ----
    per_robot_goal_reset = True  # whether to reset individual robot goals when they are reached


    # ---- Instantiate simulation environment and model ----
    sim = MARL_SIM(
        world_file="robot_nav/worlds/multi_robot_world.yaml",
        disable_plotting=True,
        reward_phase=1,
        per_robot_goal_reset=per_robot_goal_reset,
    )  # instantiate environment

    model = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        device=device,
        save_every=save_every,
        load_model=True,
        model_name="TDR-MARL-train",
        load_model_name="TDR-MARL-train",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/correct"),
        attention="igs",
    )  # instantiate a model

    # ---- Setup replay buffer and initial connections ----
    replay_buffer = get_buffer(
        model,
        sim,
        load_saved_buffer,
        pretrain,
        pretraining_iterations,
        training_iterations,
        batch_size,
    )
    connections = torch.tensor(
        [[0.0 for _ in range(sim.num_robots - 1)] for _ in range(sim.num_robots)]
    )

    # ---- Setup data saver for centralized pretraining ----
    data_saver = None
    if save_data:
        data_saver = MARLDataSaver(filepath=data_save_path)

    # ---- Take initial step in environment ----
    poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = (
        sim.step([[0, 0] for _ in range(sim.num_robots)], connections)
    )  # get the initial step state
    running_goals = 0
    running_collisions = 0
    running_timesteps = 0

    # ---- Main training loop ----
    while epoch < max_epochs:  # train until max_epochs is reached
        state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )  # get state a state representation from returned data from the environment

        action, connection, combined_weights = model.get_action(
            np.array(state), True
        )  # get an action from the model

        a_in = [
            [(a[0] + 1) / 4, a[1]] for a in action
        ]  # clip linear velocity to [0, 0.5] m/s range

        (
            poses,
            distance,
            cos,
            sin,
            collision,
            goal,
            a,
            reward,
            positions,
            goal_positions,
        ) = sim.step(
            a_in, connection, combined_weights
        )  # get data from the environment
        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_timesteps += 1
        next_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )  # get a next state representation
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )  # add experience to the replay buffer
        
        # Save data for centralized pretraining
        if data_saver is not None:
            data_saver.add(
                poses=poses,
                distances=distance,
                cos_vals=cos,
                sin_vals=sin,
                collisions=collision,
                goals=goal,
                actions=action,  # raw model output [-1, 1], NOT a_in
                goal_positions=goal_positions,
            )
        
        outside = outside_of_bounds(poses, sim)
        if (
            any(terminal) or steps == max_steps or outside or all(goal)
        ):  # reset environment of terminal state reached, or max_steps were taken
            (
                poses,
                distance,
                cos,
                sin,
                collision,
                goal,
                a,
                reward,
                positions,
                goal_positions,
            ) = sim.reset()
            episode += 1
            if episode % train_every_n == 0:
                model.writer.add_scalar(
                    "run/avg_goal", running_goals / running_timesteps, epoch
                )
                model.writer.add_scalar(
                    "run/avg_collision", running_collisions / running_timesteps, epoch
                )
                running_goals = 0
                running_collisions = 0
                running_timesteps = 0
                epoch += 1
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )  # train the model and update its parameters

            steps = 0
        else:
            steps += 1

    # ---- Save collected experience data ----
    if data_saver is not None:
        data_saver.save()
        print(f"Training complete. Data saved to {data_save_path}")


if __name__ == "__main__":
    main()
