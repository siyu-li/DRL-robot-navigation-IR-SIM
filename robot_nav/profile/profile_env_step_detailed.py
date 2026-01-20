"""
Ultra-detailed profiling script that patches the environment step method
to identify exactly what's taking time.

Usage:
    python -m robot_nav.profile_env_step_detailed
"""

from pathlib import Path
import torch
import numpy as np
import logging
import time
from collections import defaultdict
from robot_nav.SIM_ENV.marl_lidar_sim import MARL_LIDAR_SIM
from robot_nav.models.MARL.marlTD3.marlTD3_lidar import TD3WithLiDAR
from robot_nav.utils import get_buffer

# Suppress IRSim warnings
from loguru import logger
logger.disable("irsim")

# Global timing dictionary
TIMINGS = defaultdict(list)


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name):
        self.name = name
        self.start = None
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        TIMINGS[self.name].append(elapsed)


def patch_environment():
    """Patch the environment step method to add detailed timing"""
    original_step = MARL_LIDAR_SIM.step
    original_get_lidar = MARL_LIDAR_SIM.get_lidar_scans
    
    def timed_step(self, action, connection, combined_weights=None):
        with Timer("ENV_STEP_TOTAL"):
            with Timer("  env_physics_step"):
                self.env.step(action_id=[i for i in range(self.num_robots)], action=action)
            
            with Timer("  env_render"):
                self.env.render()
            
            robot_states = [
                [self.env.robot_list[i].state[0], self.env.robot_list[i].state[1]]
                for i in range(self.num_robots)
            ]
            
            with Timer("  get_lidar_scans"):
                lidar_scans = self.get_lidar_scans() if self.use_lidar else None
            
            poses = []
            distances = []
            coss = []
            sins = []
            collisions = []
            goals = []
            rewards = []
            positions = []
            goal_positions = []
            
            with Timer("  robot_loop_processing"):
                for i in range(self.num_robots):
                    robot_state = self.env.robot_list[i].state
                    
                    # Distance calculations
                    closest_robots = [
                        np.linalg.norm([
                            robot_states[j][0] - robot_state[0],
                            robot_states[j][1] - robot_state[1],
                        ])
                        for j in range(self.num_robots) if j != i
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
                    
                    with Timer("    reward_calculation"):
                        reward = self.get_reward(
                            goal, collision, action_i, closest_robots, distance,
                            self.reward_phase, lidar_scan
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
                    poses.append([robot_state[0].item(), robot_state[1].item(), robot_state[2].item()])
                    goal_positions.append(goal_position)
                    
                    # Visualization
                    if combined_weights is not None:
                        with Timer("    attention_visualization"):
                            i_weights = combined_weights[i].tolist()
                            weight_idx = 0
                            for j in range(self.num_robots):
                                if i == j:
                                    continue
                                weight = i_weights[weight_idx]
                                weight_idx += 1
                                other_robot_state = self.env.robot_list[j].state
                                other_pos = [other_robot_state[0].item(), other_robot_state[1].item()]
                                rx = [position[0], other_pos[0]]
                                ry = [position[1], other_pos[1]]
                                self.env.draw_trajectory(
                                    np.array([rx, ry]), refresh=True, linewidth=weight * 2
                                )
                    
                    # Reset goals individually if reached
                    if self.per_robot_goal_reset and goal:
                        with Timer("    goal_reset"):
                            self.env.robot_list[i].set_random_goal(
                                obstacle_list=self.env.obstacle_list,
                                init=True,
                                range_limits=[
                                    [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                                    [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                                ],
                            )
        
        if self.use_lidar:
            return (poses, distances, coss, sins, collisions, goals, action,
                   rewards, positions, goal_positions, lidar_scans)
        else:
            return (poses, distances, coss, sins, collisions, goals, action,
                   rewards, positions, goal_positions)
    
    def timed_get_lidar(self):
        """Time the LiDAR scanning process"""
        with Timer("    lidar_actual_scan"):
            result = original_get_lidar(self)
        return result
    
    MARL_LIDAR_SIM.step = timed_step
    MARL_LIDAR_SIM.get_lidar_scans = timed_get_lidar


def print_timing_stats():
    """Print detailed timing statistics"""
    print(f"\n{'='*90}")
    print("DETAILED ENVIRONMENT PROFILING RESULTS")
    print(f"{'='*90}")
    
    total_time = sum(sum(times) for times in TIMINGS.values())
    
    # Sort by total time
    sorted_items = sorted(
        [(name, times) for name, times in TIMINGS.items()],
        key=lambda x: sum(x[1]),
        reverse=True
    )
    
    print(f"{'Operation':<45} {'Count':<8} {'Total (s)':<12} {'Mean (ms)':<12} {'% of Total':<12}")
    print(f"{'-'*90}")
    
    for name, times in sorted_items:
        count = len(times)
        total = sum(times)
        mean = (total / count) * 1000 if count > 0 else 0
        percent = (total / total_time * 100) if total_time > 0 else 0
        print(f"{name:<45} {count:<8} {total:<12.4f} {mean:<12.4f} {percent:<12.2f}")
    
    print(f"{'-'*90}")
    print(f"{'TOTAL':<45} {'':<8} {total_time:<12.4f} {'':<12} {'100.00':<12}")
    print(f"{'='*90}\n")


def outside_of_bounds(poses, sim):
    """Check if any robot is outside boundaries."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def main():
    """Main profiling function - runs a short training loop"""
    
    # Patch the environment
    patch_environment()
    
    # Setup
    action_dim = 2
    max_action = 1
    state_dim = 11
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    max_steps_total = 100  # Run only 100 steps for quick profiling
    steps_count = 0
    
    # Environment parameters
    per_robot_goal_reset = True
    use_lidar = True
    lidar_num_beams = 180
    lidar_range_max = 7.0
    lidar_num_sectors = 12
    lidar_embed_dim = 12
    random_obstacles = True
    num_obstacles = 5

    print("Initializing environment...")
    sim = MARL_LIDAR_SIM(
        world_file="robot_nav/worlds/multi_robot_world_lidar.yaml",
        disable_plotting=True,
        reward_phase=1,
        use_lidar=use_lidar,
        lidar_num_beams=lidar_num_beams,
        lidar_range_max=lidar_range_max,
        random_obstacles=random_obstacles,
        num_obstacles=num_obstacles,
        per_robot_goal_reset=per_robot_goal_reset,
    )

    print("Initializing model...")
    model = TD3WithLiDAR(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        device=device,
        save_every=999999,  # Don't save
        load_model=False,
        model_name="PROFILE-TEST",
        attention="igs",
        use_lidar=use_lidar,
        lidar_encoder_type="sector",
        lidar_num_beams=lidar_num_beams,
        lidar_embed_dim=lidar_embed_dim,
        lidar_encoder_kwargs={
            "num_sectors": lidar_num_sectors,
            "aggregation": "min",
            "learnable": False,
        },
        lidar_range_max=lidar_range_max,
    )

    connections = torch.tensor(
        [[0.0 for _ in range(sim.num_robots - 1)] for _ in range(sim.num_robots)]
    )

    # Take initial step
    print(f"Running {max_steps_total} steps to profile environment...")
    (poses, distance, cos, sin, collision, goal, a, reward,
     positions, goal_positions, lidar_scans) = sim.step(
        [[0, 0] for _ in range(sim.num_robots)], connections
    )
    
    # Run simulation loop
    while steps_count < max_steps_total:
        state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions, lidar_scans
        )
        
        action, connection, combined_weights = model.get_action(np.array(state), True)
        a_in = [[(a[0] + 1) / 4, a[1]] for a in action]
        
        # This is what we're profiling
        (poses, distance, cos, sin, collision, goal, a, reward,
         positions, goal_positions, lidar_scans) = sim.step(
            a_in, connection, None  # Don't visualize attention weights
        )
        
        steps_count += 1
        
        outside = outside_of_bounds(poses, sim)
        if any(terminal) or outside or all(goal):
            (poses, distance, cos, sin, collision, goal, a, reward,
             positions, goal_positions, lidar_scans) = sim.reset(
                random_obstacles=random_obstacles
            )
        
        if steps_count % 25 == 0:
            print(f"  Completed {steps_count}/{max_steps_total} steps...")
    
    # Print results
    print_timing_stats()
    
    # Calculate breakdown
    print("\n" + "="*90)
    print("BREAKDOWN ANALYSIS")
    print("="*90)
    
    total_env = sum(TIMINGS.get("ENV_STEP_TOTAL", [0]))
    physics = sum(TIMINGS.get("  env_physics_step", [0]))
    render = sum(TIMINGS.get("  env_render", [0]))
    lidar = sum(TIMINGS.get("  get_lidar_scans", [0]))
    robot_loop = sum(TIMINGS.get("  robot_loop_processing", [0]))
    
    print(f"Total environment step time: {total_env:.4f}s")
    print(f"  - Physics simulation: {physics:.4f}s ({physics/total_env*100:.1f}%)")
    print(f"  - Rendering: {render:.4f}s ({render/total_env*100:.1f}%)")
    print(f"  - LiDAR scanning: {lidar:.4f}s ({lidar/total_env*100:.1f}%)")
    print(f"  - Robot processing loop: {robot_loop:.4f}s ({robot_loop/total_env*100:.1f}%)")
    print("="*90)


if __name__ == "__main__":
    main()
