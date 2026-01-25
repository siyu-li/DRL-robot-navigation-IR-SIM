import statistics
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from robot_nav.models.MARL.marlTD3.marlTD3_lidar import TD3WithLiDAR
import torch
import numpy as np
from robot_nav.SIM_ENV.marl_lidar_sim import MARL_LIDAR_SIM
from robot_nav.utils import MARLDataSaver

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
    """Main testing function"""

    # ---- Hyperparameters and setup ----
    action_dim = 2
    max_action = 1
    state_dim = 11
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 1
    episode = 0
    max_steps = 100
    steps = 0
    save_every = 5
    test_scenarios = 200
    save_data = False
    data_save_path = "robot_nav/assets/marl_test_data_lidar.yml"

    # ---- Instantiate simulation environment and model ----
    sim = MARL_LIDAR_SIM(
        world_file="robot_nav/worlds/multi_robot_world_lidar.yaml",
        disable_plotting=False,
        reward_phase=4,
        use_lidar=True,
        lidar_num_beams=180,
        lidar_range_max=7.0,
        random_obstacles=True,
        num_obstacles=5,
        per_robot_goal_reset=True,
    )

    model = TD3WithLiDAR(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        device=device,
        save_every=save_every,
        load_model=True,
        model_name="MARL-LiDAR-test-finetune",
        load_model_name="MARL-LiDAR-train-finetune",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/Jan.20_obstacle_finetune"),
        attention="igs",
        use_lidar=True,
        lidar_encoder_type="sector",
        lidar_num_beams=180,
        lidar_embed_dim=12,
        lidar_encoder_kwargs={
            "num_sectors": 12,
            "aggregation": "min",
            "learnable": False,
        },
        lidar_range_max=7.0,
    )

    connections = torch.tensor(
        [[0.0 for _ in range(sim.num_robots - 1)] for _ in range(sim.num_robots)]
    )

    data_saver = None
    if save_data:
        data_saver = MARLDataSaver(filepath=data_save_path)

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
    ) = sim.step([[0, 0] for _ in range(sim.num_robots)], connections)
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
    while episode < test_scenarios:
        state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions, lidar_scans
        )
        action, connection, combined_weights = model.get_action(
            np.array(state), False
        )

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

        a_in = [
            [(a[0] + 1) / 4, a[1]] for a in action
        ]

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
        )
        running_goals += sum(goal)
        running_collisions += sum(collision) / 2
        running_reward += sum(reward)
        running_timesteps += 1

        if data_saver is not None:
            data_saver.add(
                poses=poses,
                distances=distance,
                cos_vals=cos,
                sin_vals=sin,
                collisions=collision,
                goals=goal,
                actions=action,
                goal_positions=goal_positions,
                lidar_scans=lidar_scans,
            )

        for j in range(len(a_in)):
            lin_actions.append(a_in[j][0].item())
            ang_actions.append(a_in[j][1].item())
        outside = outside_of_bounds(poses, sim)

        if (
            sum(collision) > 0.5 or steps == max_steps or outside or all(goal)
        ):
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
            ) = sim.reset()
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

    reward_per_ep = np.array(reward_per_ep, dtype=np.float32)
    goals_per_ep = np.array(goals_per_ep, dtype=np.float32)
    col_per_ep = np.array(col_per_ep, dtype=np.float32)
    lin_actions = np.array(lin_actions, dtype=np.float32)
    ang_actions = np.array(ang_actions, dtype=np.float32)
    entropy = np.array(entropy_list, dtype=np.float32)
    avg_ep_reward = statistics.mean(reward_per_ep)
    avg_ep_reward_std = statistics.stdev(reward_per_ep)
    avg_ep_col = statistics.mean(col_per_ep)
    avg_ep_col_std = statistics.stdev(col_per_ep)
    avg_ep_goals = statistics.mean(goals_per_ep)
    avg_ep_goals_std = statistics.stdev(goals_per_ep)
    mean_lin_action = statistics.mean(lin_actions)
    lin_actions_std = statistics.stdev(lin_actions)
    mean_ang_action = statistics.mean(ang_actions)
    ang_actions_std = statistics.stdev(ang_actions)
    mean_entropy = statistics.mean(entropy)
    mean_entropy_std = statistics.stdev(entropy)

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

    counts, bin_edges = np.histogram(lin_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/lin_actions_hist", fig)

    counts, bin_edges = np.histogram(ang_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/ang_actions_hist", fig)

    if data_saver is not None:
        data_saver.save()
        print(f"Test complete. Data saved to {data_save_path}")

if __name__ == "__main__":
    main()