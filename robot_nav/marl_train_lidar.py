"""
MARL Training Script with LiDAR Support.

This script trains a multi-agent TD3 policy with LiDAR observations for
robot navigation with obstacle avoidance.

Usage:
    python -m robot_nav.marl_train_lidar
"""

from pathlib import Path

from robot_nav.models.MARL.marlTD3.marlTD3_lidar import TD3WithLiDAR

import torch
import numpy as np
import logging
from robot_nav.SIM_ENV.marl_lidar_sim import MARL_LIDAR_SIM
from robot_nav.utils import get_buffer

# Suppress IRSim warnings
# logging.getLogger("irsim").setLevel(logging.ERROR)
from loguru import logger
logger.disable("irsim")

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
    print(f"Using device: {device}")
    max_epochs = 500  # max number of epochs
    epoch = 1  # starting epoch number
    episode = 0  # starting episode number
    train_every_n = 10  # train and update network parameters every n episodes
    training_iterations = 80  # how many batches to use for single training cycle
    batch_size = 16  # batch size for each training iteration
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    load_saved_buffer = False  # whether to load experiences from assets/data.yml
    pretrain = False  # whether to use the loaded experiences to pre-train the model
    pretraining_iterations = 10  # number of training iterations to run during pre-training
    save_every = 5  # save the model every n training cycles

    # ---- Environment Hyperparameters ----
    per_robot_goal_reset = True  # whether to reset individual robot goals when they are reached


    # ---- LiDAR Hyperparameters ----
    use_lidar = True  # whether to use LiDAR observations
    lidar_num_beams = 180  # number of LiDAR beams
    lidar_range_max = 7.0  # maximum LiDAR range
    lidar_num_sectors = 12  # number of sectors for min-range aggregation
    lidar_embed_dim = 12  # LiDAR embedding dim = num_sectors (no NN projection)

    # ---- Obstacle Hyperparameters ----
    random_obstacles = True  # whether to use random obstacles
    num_obstacles = 5  # number of obstacles

    # ---- Instantiate simulation environment and model ----
    sim = MARL_LIDAR_SIM(
        world_file="robot_nav/worlds/multi_robot_world_lidar.yaml",
        disable_plotting=True,
        reward_phase=4,
        use_lidar=use_lidar,
        lidar_num_beams=lidar_num_beams,
        lidar_range_max=lidar_range_max,
        random_obstacles=random_obstacles,
        num_obstacles=num_obstacles,
        per_robot_goal_reset=per_robot_goal_reset,
    )  # instantiate environment

    model = TD3WithLiDAR(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        device=device,
        save_every=save_every,
        load_model=True,
        save_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/Jan.24_obstacle_finetune"),
        model_name="MARL-LiDAR-train-finetune-reward4",
        load_model_name="MARL-LiDAR-train",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/Jan.19_obstacle_ver2"),
        attention="igs",
        # Load pretrained attention weights from decentralized model
        load_pretrained_attention=False,
        pretrained_attention_model_name="TDR-MARL-train",
        pretrained_attention_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/correct"),
        freeze_attention=True,  # Set to True to freeze attention during training
        use_lidar=use_lidar,
        lidar_encoder_type="sector",  # use sector encoder (min-range, no NN)
        lidar_num_beams=lidar_num_beams,
        lidar_embed_dim=lidar_embed_dim,
        lidar_encoder_kwargs={
            "num_sectors": lidar_num_sectors,
            "aggregation": "min",
            "learnable": False,
        },
        lidar_range_max=lidar_range_max,
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

    # ---- Take initial step in environment ----
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
        lidar_scans,
    ) = sim.step(
        [[0, 0] for _ in range(sim.num_robots)], connections
    )  # get the initial step state
    running_goals = 0
    running_collisions = 0
    running_timesteps = 0

    # ---- Main training loop ----
    while epoch < max_epochs:  # train until max_epochs is reached
        state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions, lidar_scans
        )  # get state representation from returned data from the environment

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
            lidar_scans,
        ) = sim.step(
            a_in, connection, combined_weights
        )  # get data from the environment
        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_timesteps += 1
        next_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions, lidar_scans
        )  # get a next state representation
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )  # add experience to the replay buffer

        outside = outside_of_bounds(poses, sim)
        if (
            any(terminal) or steps == max_steps or outside or all(goal)
        ):  # reset environment if terminal state reached, or max_steps were taken
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
                lidar_scans,
            ) = sim.reset(random_obstacles=random_obstacles)
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

    print("Training complete.")


if __name__ == "__main__":
    main()
