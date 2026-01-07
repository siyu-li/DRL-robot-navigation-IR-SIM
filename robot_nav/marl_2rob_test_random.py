import statistics
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from robot_nav.models.MARL.marlTD3.marlTD3_centralized_2robot import marlTD3_2robot
import torch
import numpy as np
from robot_nav.SIM_ENV.marl_centralized_2robot_sim import MARL_2Robot_SIM


def outside_of_bounds(poses, x_range=(0, 12), y_range=(0, 12)):
    """
    Check if any robot is outside the defined world boundaries.

    Args:
        poses (list): List of [x, y, theta] poses for each robot.
        x_range (tuple): Min and max x coordinates.
        y_range (tuple): Min and max y coordinates.

    Returns:
        bool: True if any robot is outside the bounds, else False.
    """
    for pose in poses:
        if pose[0] < x_range[0] or pose[0] > x_range[1]:
            return True
        if pose[1] < y_range[0] or pose[1] > y_range[1]:
            return True
    return False


def main(args=None):
    """Main testing function for 2-robot centralized control."""

    # ---- Hyperparameters and setup ----
    state_dim = 11  # number of input values in the neural network
    num_total_robots = 5
    active_robot_ids = [0, 1]  # Which 2 robots to control
    coupled_mode = False  # True = shared linear velocity, False = independent
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    epoch = 1  # starting epoch number
    episode = 0
    max_steps = 600  # maximum number of steps in single episode
    steps = 0  # starting step number
    save_every = 5  # save the model every n training cycles
    test_scenarios = 100

    # ---- Instantiate simulation environment and model ----
    sim = MARL_2Robot_SIM(
        world_file="robot_nav/worlds/multi_robot_world.yaml",
        disable_plotting=False,
        reward_phase=2,
        active_robot_ids=active_robot_ids,
        coupled_mode=coupled_mode,
    )  # instantiate environment

    model = marlTD3_2robot(
        state_dim=state_dim,
        num_total_robots=num_total_robots,
        active_robot_ids=active_robot_ids,
        device=device,
        coupled_mode=coupled_mode,
        save_every=save_every,
        load_model=True,
        model_name="TDR-MARL-2robot-test",
        load_model_name="TDR-MARL-2robot-train",
        load_directory=Path("robot_nav/models/MARL/marlTD3_centralized_2robot/checkpoint"),
        attention="igs",
        # Load pretrained attention from decentralized model
        load_pretrained_attention=True,
        pretrained_attention_model_name="TDR-MARL-train",
        pretrained_attention_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        freeze_attention=True,  # Freeze attention, only train policy
    )  # instantiate a model

    # ---- Take initial step in environment ----
    (
        active_poses, active_distances, active_coss, active_sins,
        active_collisions, active_goals, a, active_rewards,
        active_positions, active_goal_positions, all_poses, _
    ) = sim.reset()  # get the initial step state

    running_goals = 0
    running_reward = 0
    running_collisions = 0
    running_timesteps = 0

    reward_per_ep = []
    goals_per_ep = []
    col_per_ep = []
    lin_actions = []
    ang_actions = []
    entropy_list = []
    epsilon = 1e-6
    pbar = tqdm(total=test_scenarios)
    
    # Initialize previous action (scaled action format used in prepare_state)
    prev_action = [[0, 0], [0, 0]] if not coupled_mode else [0, 0, 0]

    while episode < test_scenarios:
        # Prepare state for all robots (for attention network)
        # Use the previous action from the last step
        state = model.prepare_state(
            all_poses, active_distances, active_coss, active_sins,
            active_collisions, prev_action, active_goal_positions
        )

        # Get action for 2 active robots (no noise during testing)
        action, connection, combined_weights = model.get_action(
            np.array(state), add_noise=False
        )

        # Calculate entropy for analysis/logging
        if combined_weights is not None:
            combined_weights_norm = combined_weights / (
                combined_weights.sum(dim=-1, keepdim=True) + epsilon
            )
            entropy = (
                (
                    -(combined_weights_norm * (combined_weights_norm + epsilon).log())
                    .sum(dim=-1)
                    .mean()
                )
                .data.cpu()
                .numpy()
            )
            entropy_list.append(entropy)

        # Convert action to environment format
        if coupled_mode:
            a_in = action  # [lin, ang1, ang2]
            a_in_scaled = [(action[0] + 1) / 4, action[1], action[2]]
            prev_action = a_in_scaled  # Save for next state preparation
        else:
            a_in = [[(a[0] + 1) / 4, a[1]] for a in action]
            prev_action = a_in  # Save for next state preparation

        # Step environment
        (
            active_poses, active_distances, active_coss, active_sins,
            active_collisions, active_goals, a, active_rewards,
            active_positions, active_goal_positions, all_poses, episode_done
        ) = sim.step(a_in if not coupled_mode else a_in_scaled)

        running_goals += sum(active_goals)
        running_collisions += sum(active_collisions) / 2
        running_reward += sum(active_rewards)
        running_timesteps += 1

        # Record actions for statistics
        if coupled_mode:
            lin_actions.append(a_in_scaled[0])
            ang_actions.append(a_in_scaled[1])
            ang_actions.append(a_in_scaled[2])
        else:
            for j in range(len(a_in)):
                lin_actions.append(a_in[j][0])
                ang_actions.append(a_in[j][1])

        outside = outside_of_bounds(active_poses)

        # Reset environment if terminal state reached, or max_steps were taken
        if (
            sum(active_collisions) > 0.5 or steps == max_steps or outside or episode_done
        ):
            (
                active_poses, active_distances, active_coss, active_sins,
                active_collisions, active_goals, a, active_rewards,
                active_positions, active_goal_positions, all_poses, _
            ) = sim.reset()
            
            # Reset previous action for the new episode
            prev_action = [[0, 0], [0, 0]] if not coupled_mode else [0, 0, 0]
            
            reward_per_ep.append(running_reward)
            running_reward = 0
            goals_per_ep.append(running_goals)
            running_goals = 0
            col_per_ep.append(running_collisions)
            running_collisions = 0

            steps = 0
            episode += 1
            pbar.update(1)
        else:
            steps += 1

    pbar.close()

    # ---- Compute statistics ----
    reward_per_ep = np.array(reward_per_ep, dtype=np.float32)
    goals_per_ep = np.array(goals_per_ep, dtype=np.float32)
    col_per_ep = np.array(col_per_ep, dtype=np.float32)
    lin_actions = np.array(lin_actions, dtype=np.float32)
    ang_actions = np.array(ang_actions, dtype=np.float32)
    
    avg_ep_reward = statistics.mean(reward_per_ep)
    avg_ep_reward_std = statistics.stdev(reward_per_ep) if len(reward_per_ep) > 1 else 0
    avg_ep_col = statistics.mean(col_per_ep)
    avg_ep_col_std = statistics.stdev(col_per_ep) if len(col_per_ep) > 1 else 0
    avg_ep_goals = statistics.mean(goals_per_ep)
    avg_ep_goals_std = statistics.stdev(goals_per_ep) if len(goals_per_ep) > 1 else 0
    mean_lin_action = statistics.mean(lin_actions)
    lin_actions_std = statistics.stdev(lin_actions) if len(lin_actions) > 1 else 0
    mean_ang_action = statistics.mean(ang_actions)
    ang_actions_std = statistics.stdev(ang_actions) if len(ang_actions) > 1 else 0

    if len(entropy_list) > 1:
        entropy = np.array(entropy_list, dtype=np.float32)
        mean_entropy = statistics.mean(entropy)
        mean_entropy_std = statistics.stdev(entropy)
    else:
        mean_entropy = 0
        mean_entropy_std = 0

    # ---- Print results ----
    print(f"avg_ep_reward: {avg_ep_reward}")
    print(f"avg_ep_reward_std: {avg_ep_reward_std}")
    print(f"avg_ep_col: {avg_ep_col}")
    print(f"avg_ep_col_std: {avg_ep_col_std}")
    print(f"avg_ep_goals: {avg_ep_goals}")
    print(f"avg_ep_goals_std: {avg_ep_goals_std}")
    print(f"mean_lin_action: {mean_lin_action}")
    print(f"lin_actions_std: {lin_actions_std}")
    print(f"mean_ang_action: {mean_ang_action}")
    print(f"ang_actions_std: {ang_actions_std}")
    print(f"mean_entropy: {mean_entropy}")
    print(f"mean_entropy_std: {mean_entropy_std}")
    print("..............................................")

    # ---- Log to TensorBoard ----
    model.writer.add_scalar("test/avg_ep_reward", avg_ep_reward, epoch)
    model.writer.add_scalar("test/avg_ep_reward_std", avg_ep_reward_std, epoch)
    model.writer.add_scalar("test/avg_ep_col", avg_ep_col, epoch)
    model.writer.add_scalar("test/avg_ep_col_std", avg_ep_col_std, epoch)
    model.writer.add_scalar("test/avg_ep_goals", avg_ep_goals, epoch)
    model.writer.add_scalar("test/avg_ep_goals_std", avg_ep_goals_std, epoch)
    model.writer.add_scalar("test/mean_lin_action", mean_lin_action, epoch)
    model.writer.add_scalar("test/lin_actions_std", lin_actions_std, epoch)
    model.writer.add_scalar("test/mean_ang_action", mean_ang_action, epoch)
    model.writer.add_scalar("test/ang_actions_std", ang_actions_std, epoch)
    model.writer.add_scalar("test/mean_entropy", mean_entropy, epoch)
    model.writer.add_scalar("test/mean_entropy_std", mean_entropy_std, epoch)

    bins = 100
    model.writer.add_histogram("test/lin_actions", lin_actions, epoch, max_bins=bins)
    model.writer.add_histogram("test/ang_actions", ang_actions, epoch, max_bins=bins)

    # ---- Create histograms ----
    counts, bin_edges = np.histogram(lin_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Linear Actions Histogram (2-Robot Test)")
    model.writer.add_figure("test/lin_actions_hist", fig)

    counts, bin_edges = np.histogram(ang_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Angular Actions Histogram (2-Robot Test)")
    model.writer.add_figure("test/ang_actions_hist", fig)


if __name__ == "__main__":
    main()
