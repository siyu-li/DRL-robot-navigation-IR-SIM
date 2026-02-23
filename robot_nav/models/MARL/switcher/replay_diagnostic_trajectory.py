"""
Replay a recorded diagnostic trajectory in the simulation with rendering.

Loads a diagnostic_trajectories.pt file and replays a specific episode
by restoring the initial configuration (robot poses, goals, obstacle
positions) and then applying the exact recorded Phase-2 actions at each
step. The simulation renders live so you can visually inspect what
happens to stuck / timed-out robots.

Usage:
    python -m robot_nav.models.MARL.switcher.replay_diagnostic_trajectory \
        --episode 15 [--data PATH] [--speed 1.0] [--pause-at-step N]

Keys during replay (when paused):
    Enter  — advance one step
    c      — continue (unpause)
    q      — quit
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from loguru import logger as loguru_logger
loguru_logger.disable("irsim")


def parse_args():
    p = argparse.ArgumentParser(description="Replay a diagnostic trajectory")
    p.add_argument("--episode", type=int, default=15,
                   help="Episode index to replay (default: 15)")
    p.add_argument("--data", type=str,
                   default="robot_nav/models/MARL/switcher/data/diagnostic_trajectories.pt",
                   help="Path to diagnostic_trajectories.pt")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0, 0.5=slow, 2.0=fast)")
    p.add_argument("--pause-at-step", type=int, default=None,
                   help="Pause replay at this episode step (for inspection)")
    p.add_argument("--world", type=str,
                   default="robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
                   help="World YAML file")
    p.add_argument("--step-mode", action="store_true",
                   help="Step-by-step mode: press Enter to advance each macro step")
    return p.parse_args()


def obstacle_states_to_raw(obs_states: np.ndarray) -> List[np.ndarray]:
    """
    Convert obstacle states from (N_obs, 4) [x, y, cos_h, sin_h]
    to list of (3,1) arrays [[x],[y],[theta]] for set_state.
    """
    result = []
    for i in range(obs_states.shape[0]):
        x = obs_states[i, 0]
        y = obs_states[i, 1]
        cos_h = obs_states[i, 2]
        sin_h = obs_states[i, 3]
        theta = math.atan2(sin_h, cos_h)
        result.append(np.array([[x], [y], [theta]]))
    return result


def print_episode_summary(ep: Dict, groups: List):
    """Print a detailed summary of the episode before replay."""
    N = 14
    print("\n" + "=" * 78)
    print(f"  EPISODE {ep['episode_id']}  —  {ep['outcome'].upper()}")
    print("=" * 78)
    print(f"  Total sim steps:  {ep['total_sim_steps']}")
    print(f"  Robots reached:   {ep['n_reached']}/{N}")
    print(f"  Never reached:    {ep['never_reached_robots']}")
    print(f"  Step records:     {len(ep['steps'])}")

    # Per-robot reach times
    print(f"\n  Per-robot reach step:")
    for i in range(N):
        rs = ep['per_robot_reach_step'][i]
        marker = "  *** NEVER ***" if rs == -1 else ""
        print(f"    Robot {i:>2}: step {rs:>5}{marker}")

    # Initial distances
    ic = ep['initial_config']
    print(f"\n  Initial distances to goal:")
    for i in range(N):
        d = ic['distances'][i]
        marker = " <-- stuck" if i in ep['never_reached_robots'] else ""
        print(f"    Robot {i:>2}: {d:>7.3f}{marker}")

    # Stuck robot analysis
    stuck = ep['never_reached_robots']
    if stuck:
        print(f"\n  --- Stuck robot analysis ---")
        steps = ep['steps']
        for ri in stuck:
            # Velocity stats
            lv = [s['raw_lin_vel'][ri] for s in steps]
            av = [s['raw_ang_vel'][ri] for s in steps]
            disp = [s['per_robot_displacement'][ri] for s in steps]
            n_in_sel = sum(1 for s in steps if ri in s['selected_group'])
            n_singleton = sum(1 for s in steps if s['selected_group'] == [ri])
            n_urgent = sum(1 for s in steps if s['urgency_flags'][ri])

            # Distance over time (sample a few)
            d_samples = [steps[t]['post_distances'][ri]
                         for t in range(0, len(steps), max(1, len(steps) // 10))]

            print(f"\n    Robot {ri}:")
            print(f"      Raw lin_vel:  mean={np.mean(lv):.5f}  max={np.max(lv):.5f}  "
                  f"near-zero (<0.02): {sum(1 for v in lv if v < 0.02)}/{len(lv)}")
            print(f"      Raw ang_vel:  mean={np.mean(av):.4f}")
            print(f"      Displacement: mean={np.mean(disp):.5f}  max={np.max(disp):.5f}")
            print(f"      In selected group: {n_in_sel}/{len(steps)} steps")
            print(f"      As singleton:      {n_singleton}/{len(steps)} steps")
            print(f"      Urgency flagged:   {n_urgent}/{len(steps)} steps")
            print(f"      Distance samples:  {['%.3f' % d for d in d_samples]}")
            print(f"      Final distance:    {steps[-1]['post_distances'][ri]:.3f}")

    print("=" * 78)


def replay_episode(ep: Dict, groups: List, args):
    """Replay one episode with live rendering."""
    from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE

    N = 14
    ic = ep['initial_config']
    steps = ep['steps']

    # Create sim with rendering ON
    sim = MARL_SIM_OBSTACLE(
        world_file=args.world,
        disable_plotting=False,  # rendering on!
        reward_phase=5,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=1.5,
    )

    # === Restore initial configuration ===
    # Use reset() with obstacle_states parameter to ensure proper initialization
    
    import sys
    print("\n=== OBSTACLE POSITIONS ===", flush=True)
    print("Saved obstacle states from episode data:", flush=True)
    for oi in range(ic['obstacle_states'].shape[0]):
        x, y = ic['obstacle_states'][oi, 0], ic['obstacle_states'][oi, 1]
        cos_h, sin_h = ic['obstacle_states'][oi, 2], ic['obstacle_states'][oi, 3]
        theta = np.arctan2(sin_h, cos_h)
        print(f"  Obs {oi}: x={x:.3f}, y={y:.3f}, theta={theta:.3f}", flush=True)
    
    # Convert obstacle states to raw format for reset
    obs_raw = obstacle_states_to_raw(ic['obstacle_states'])
    
    # Call reset with obstacle_states parameter (this handles env.reset() internally)
    # Note: We still need to set robot states after reset since reset randomizes them
    sim.reset(obstacle_states=obs_raw)
    
    # Now set robot positions and goals (after reset)
    for ri, robot in enumerate(sim.env.robot_list):
        pose = ic['poses'][ri]
        robot.set_state(
            state=np.array([[pose[0]], [pose[1]], [pose[2]]]),
            init=True,
        )
        goal = ic['goal_positions'][ri]
        robot.set_goal(np.array([[goal[0]], [goal[1]], [0.0]]), init=True)
    
    # Rebuild tree after setting robot states
    sim.env.build_tree()
    sim.env._objects_check_status()
    
    print("\nObstacle positions AFTER reset with obstacle_states:", flush=True)
    for oi, obs in enumerate(sim.env.obstacle_list):
        pos = obs.position.flatten()
        state = obs.state.flatten()
        print(f"  Obs {oi}: position=({pos[0]:.3f}, {pos[1]:.3f}), state=({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f})", flush=True)
    
    # Render
    sim.env.render()
    
    print("\n=== Replay starting - obstacles set via reset() ===\n", flush=True)

    # Timing
    step_time = sim.env._world.step_time if hasattr(sim.env._world, 'step_time') else 0.3
    base_delay = step_time / args.speed

    reached = [False] * N
    goal_threshold = 0.3
    paused = args.step_mode

    print(f"\n  Starting replay... ({len(steps)} macro steps, "
          f"{sum(len(s['phase2_actions']) for s in steps)} sub-steps)")
    print(f"  Speed: {args.speed}x  |  Step mode: {'ON' if paused else 'OFF'}")
    if args.pause_at_step is not None:
        print(f"  Will pause at episode step {args.pause_at_step}")
    print()

    for macro_idx, step_rec in enumerate(steps):
        selected_group = step_rec['selected_group']
        phase2_actions = step_rec['phase2_actions']
        ep_step = step_rec['step_in_episode']

        # Print macro step info
        n_reached = sum(reached)
        stuck_info = ""
        for ri in ep['never_reached_robots']:
            if not reached[ri]:
                stuck_info += f"  R{ri}:lv={step_rec['raw_lin_vel'][ri]:.4f}"

        grp_str = str(selected_group)
        if len(grp_str) > 20:
            grp_str = grp_str[:20] + "..."
        print(f"  [{macro_idx+1:>3}/{len(steps)}] ep_step={ep_step:>5}  "
              f"grp={grp_str:<22}  score={step_rec['selected_score']:>+7.2f}  "
              f"reached={n_reached}/{N}{stuck_info}")

        # Check if we should pause
        if args.pause_at_step is not None and ep_step >= args.pause_at_step:
            paused = True

        if paused:
            try:
                user = input("    [PAUSED] Enter=step, c=continue, q=quit: ").strip().lower()
                if user == 'q':
                    print("  Quitting replay.")
                    return
                elif user == 'c':
                    paused = False
            except (KeyboardInterrupt, EOFError):
                print("\n  Interrupted.")
                return

        # Execute each sub-step of Phase 2
        for sub_idx, act_rec in enumerate(phase2_actions):
            applied = act_rec['applied']  # list of [lin_vel, ang_vel] per robot

            # Step the sim with recorded actions
            (poses, distance, cos_, sin_, collision, goals,
             action, reward, positions, goal_positions, obstacle_states
             ) = sim.step(applied, None, None)

            # Update reached (for display only — the recorded data is authoritative)
            for ri in range(N):
                if not reached[ri] and distance[ri] < goal_threshold:
                    reached[ri] = True
                    print(f"    *** Robot {ri} reached goal at ep_step ~{ep_step} ***")

            # Delay for visualization
            time.sleep(base_delay)

        # Check for collisions (just report, don't stop)
        if any(collision):
            collided = [ri for ri in range(N) if collision[ri]]
            print(f"    !!! Collision: robots {collided}")

    # Final state
    print(f"\n  Replay complete.")
    print(f"  Outcome: {ep['outcome']}")
    print(f"  Final reached: {sum(reached)}/{N}")
    if ep['never_reached_robots']:
        print(f"  Never reached: {ep['never_reached_robots']}")
        for ri in ep['never_reached_robots']:
            last = steps[-1]
            print(f"    Robot {ri}: final_dist={last['post_distances'][ri]:.3f}  "
                  f"final_pos=({last['post_poses'][ri][0]:.2f}, {last['post_poses'][ri][1]:.2f})")

    # Keep window open
    print("\n  Press Enter to close...")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass


def main():
    args = parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading {data_path}...")
    data = torch.load(data_path, weights_only=False)
    episodes = data['episodes']
    groups = data['groups']

    print(f"Loaded {len(episodes)} episodes")

    # Find the requested episode
    ep = None
    for e in episodes:
        if e['episode_id'] == args.episode:
            ep = e
            break

    if ep is None:
        print(f"Error: episode {args.episode} not found.")
        print(f"Available episodes: {[e['episode_id'] for e in episodes]}")
        sys.exit(1)

    # Print summary
    print_episode_summary(ep, groups)

    # Replay
    replay_episode(ep, groups, args)


if __name__ == "__main__":
    main()
