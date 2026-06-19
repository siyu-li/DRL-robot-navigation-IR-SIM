"""
Coarse-only test simulation for the 6-robot obstacle world.

Drives the swarm using *only* CoarseSteering (no GAT / TD3 policy and no learned
switcher).  Since there is no switcher yet, the move group is chosen at random
from the selectable groups at every coarse decision.

Each coarse decision executes one coarse control of the chosen group:
    1. a sequence of rotation sub-steps (A-matrix steering, fully realised), then
    2. a single forward move sub-step (only the chosen group's members move).

Runs both steering methods (least_squares and nonlinear) on the same env setup
as ``marl_test_obstacle_6robots.py`` and reports success rate, reward, a rough
physical-efficiency ratio, and a runtime comparison of the two solvers.

Usage:
    python -m robot_nav.marl_test_coarse_6robots
"""

import random
import time

import numpy as np
from loguru import logger
from shapely.geometry import Point

from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering

# Suppress IRSim warnings - irsim uses loguru, not standard logging
logger.disable("irsim")


def robot_outside_bounds(pose, sim):
    """Check if a specific robot is outside world boundaries."""
    if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
        return True
    if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
        return True
    return False


def build_states(poses, goal_positions):
    """
    Build the (N, 11) per-robot state CoarseSteering consumes.

    Mirrors the relevant columns of ``TD3Obstacle.prepare_state`` (positions,
    heading cos/sin, goal).  Columns CoarseSteering does not read are filled
    with zeros — no learned model is involved here.
    """
    states = []
    for pose, goal in zip(poses, goal_positions):
        px, py, theta = pose
        gx, gy = goal
        states.append(
            [px, py, np.cos(theta), np.sin(theta), 0.0, 0.0, 0.0, 0.0, 0.0, gx, gy]
        )
    return np.array(states, dtype=np.float64)


def reset_single_robot(sim, robot_idx, init_states):
    """Reset a single robot to a new collision-free random position and goal."""
    x_min, x_max = sim.x_range[0] + 1, sim.x_range[1] - 1
    y_min, y_max = sim.y_range[0] + 1, sim.y_range[1] - 1
    robot = sim.env.robot_list[robot_idx]

    conflict = True
    while conflict:
        conflict = False
        robot_state = [
            [random.uniform(x_min, x_max)],
            [random.uniform(y_min, y_max)],
            [random.uniform(-np.pi, np.pi)],
        ]
        pos = [robot_state[0][0], robot_state[1][0]]
        for j, loc in enumerate(init_states):
            if j == robot_idx:
                continue
            if np.linalg.norm([pos[0] - loc[0], pos[1] - loc[1]]) < 0.6:
                conflict = True
                break
        if not conflict:
            robot_point = Point(pos[0], pos[1])
            for obs in sim.env.obstacle_list:
                if obs.geometry.distance(robot_point) < 0.5:
                    conflict = True
                    break

    init_states[robot_idx] = pos
    robot.set_state(state=np.array(robot_state), init=True)
    robot.set_random_goal(
        obstacle_list=sim.env.obstacle_list,
        init=True,
        range_limits=[
            [sim.x_range[0] + 1, sim.y_range[0] + 1, -np.pi],
            [sim.x_range[1] - 1, sim.y_range[1] - 1, np.pi],
        ],
    )
    return init_states


def run_eval(
    sim,
    method,
    max_episodes,
    max_steps,
    render_delay,
    lin_vel=0.25,
    seed=0,
):
    """
    Run coarse-only evaluation for a single steering method.

    Returns:
        stats (dict): success rate, reward and runtime summary.
    """
    coarse = CoarseSteering(
        num_robots=sim.num_robots,
        lin_vel=lin_vel,
        method=method,
        step_time=sim.env.step_time,
        ang_max=1.0,
        seed=seed,
    )
    selectable = coarse.selectable_groups()

    completed = []            # (robot_idx, 'goal'|'collision', traj_reward)
    traj_reward = [0.0] * sim.num_robots
    path_len = [0.0] * sim.num_robots       # distance travelled this trajectory
    straight_line = [0.0] * sim.num_robots  # start→goal distance this trajectory
    eff_ratios = []                         # path_len / straight_line for goals

    solve_times = []
    total_sim_steps = 0
    episode_rewards = []

    print(f"\n{'='*80}\nMETHOD: {method}\n{'='*80}")

    for episode in range(max_episodes):
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, obstacle_states,
        ) = sim.reset(random_obstacles=True)

        current_positions = [[p[0], p[1]] for p in poses]
        traj_reward = [0.0] * sim.num_robots
        path_len = [0.0] * sim.num_robots
        straight_line = [
            float(np.linalg.norm([gp[0] - p[0], gp[1] - p[1]]))
            for p, gp in zip(poses, goal_positions)
        ]

        episode_reward = 0.0
        steps = 0

        while steps < max_steps:
            # ---- One coarse decision: random selectable group --------------
            group = random.choice(selectable)
            state = build_states(poses, goal_positions)

            rotation_frames, move_frame = coarse.compute_actions(state, group)
            solve_times.append(coarse.last_solve_time_s)

            # Rotation sub-steps followed by the single move sub-step.
            decision_frames = rotation_frames + [move_frame]

            terminated_episode = False
            for frame in decision_frames:
                prev_positions = [[p[0], p[1]] for p in poses]

                (
                    poses, distance, cos, sin, collision, goal, a, reward,
                    positions, goal_positions, obstacle_states,
                ) = sim.step(frame, None, None)

                steps += 1
                total_sim_steps += 1
                episode_reward += float(np.sum(reward))

                # ---- Per-robot bookkeeping ---------------------------------
                for i in range(sim.num_robots):
                    traj_reward[i] += reward[i]
                    path_len[i] += float(
                        np.linalg.norm(
                            [poses[i][0] - prev_positions[i][0],
                             poses[i][1] - prev_positions[i][1]]
                        )
                    )

                # ---- Trajectory completion + respawn -----------------------
                to_reset = []
                for i in range(sim.num_robots):
                    if collision[i] or robot_outside_bounds(poses[i], sim):
                        completed.append((i, 'collision', traj_reward[i]))
                        to_reset.append(i)
                    elif goal[i]:
                        completed.append((i, 'goal', traj_reward[i]))
                        if straight_line[i] > 1e-6:
                            eff_ratios.append(path_len[i] / straight_line[i])
                        to_reset.append(i)

                current_positions = [[p[0], p[1]] for p in poses]
                for ridx in to_reset:
                    current_positions = reset_single_robot(
                        sim, ridx, current_positions
                    )
                    traj_reward[ridx] = 0.0
                    path_len[ridx] = 0.0
                    # New trajectory straight-line distance after respawn.
                    new_pose = poses[ridx]
                    new_goal = sim.env.robot_list[ridx].goal
                    straight_line[ridx] = float(
                        np.linalg.norm(
                            [new_goal[0] - new_pose[0], new_goal[1] - new_pose[1]]
                        )
                    )

                if render_delay:
                    time.sleep(render_delay)

                if steps >= max_steps:
                    terminated_episode = True
                    break

            if terminated_episode:
                break

        episode_rewards.append(episode_reward)
        print(
            f"  Episode {episode + 1}/{max_episodes} | "
            f"reward {episode_reward:+.1f} | trajectories so far {len(completed)}"
        )

    # ---- Aggregate -----------------------------------------------------
    total = len(completed)
    success = sum(1 for t in completed if t[1] == 'goal')
    success_rate = (success / total * 100) if total else 0.0
    solve_arr = np.array(solve_times) if solve_times else np.array([0.0])

    stats = {
        "method": method,
        "success_rate": success_rate,
        "trajectories": total,
        "successes": success,
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_efficiency": float(np.mean(eff_ratios)) if eff_ratios else float('nan'),
        "n_solves": len(solve_times),
        "mean_solve_us": float(solve_arr.mean() * 1e6),
        "median_solve_us": float(np.median(solve_arr) * 1e6),
        "total_solve_s": float(solve_arr.sum()),
        "total_sim_steps": total_sim_steps,
    }

    print(
        f"\n  [{method}] success {success}/{total} = {success_rate:.1f}% | "
        f"mean episode reward {stats['mean_episode_reward']:+.1f} | "
        f"path/straight {stats['mean_efficiency']:.2f}\n"
        f"  [{method}] solve: {stats['n_solves']} calls | "
        f"mean {stats['mean_solve_us']:.1f} µs | "
        f"median {stats['median_solve_us']:.1f} µs | "
        f"total {stats['total_solve_s']:.3f} s"
    )
    return stats


def main():
    max_episodes = 5
    max_steps = 500
    render_delay = 0.0  # set >0 (and disable_plotting=False) to watch

    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle.yaml",
        disable_plotting=False,
        reward_phase=6,
        per_robot_goal_reset=False,  # resets handled manually
        obstacle_proximity_threshold=1.5,
    )
    print(
        f"Environment initialized: {sim.num_robots} robots, "
        f"{sim.num_obstacles} obstacles, step_time={sim.env.step_time}"
    )

    results = {}
    for method in ("least_squares", "nonlinear"):
        results[method] = run_eval(
            sim, method, max_episodes, max_steps, render_delay, seed=0
        )

    # ---- Runtime comparison -------------------------------------------
    ls, nl = results["least_squares"], results["nonlinear"]
    slowdown = (
        nl["mean_solve_us"] / ls["mean_solve_us"] if ls["mean_solve_us"] > 0 else float('nan')
    )
    print("\n" + "=" * 80)
    print("RUNTIME COMPARISON (per coarse-decision solve)")
    print("=" * 80)
    print(f"  least_squares : {ls['mean_solve_us']:8.1f} µs/call (median {ls['median_solve_us']:.1f})")
    print(f"  nonlinear     : {nl['mean_solve_us']:8.1f} µs/call (median {nl['median_solve_us']:.1f})")
    print(f"  nonlinear is {slowdown:.1f}x slower than least_squares")
    print("=" * 80)


if __name__ == "__main__":
    main()
