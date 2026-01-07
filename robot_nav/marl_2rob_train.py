import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from robot_nav.SIM_ENV.marl_centralized_2robot_sim import MARL_2Robot_SIM
from robot_nav.models.MARL.marlTD3.marlTD3_centralized_2robot import marlTD3_2robot
from robot_nav.replay_buffer import ReplayBuffer


def outside_of_bounds(poses, x_range=(0, 12), y_range=(0, 12)):
    """Check if any robot is outside bounds."""
    for pose in poses:
        if pose[0] < x_range[0] or pose[0] > x_range[1]:
            return True
        if pose[1] < y_range[0] or pose[1] > y_range[1]:
            return True
    return False


def main():
    """Main training function for 2-robot centralized control."""
    
    # ---- Hyperparameters ----
    state_dim = 11
    num_total_robots = 5
    active_robot_ids = [0, 1]  # Which 2 robots to control
    coupled_mode = False  # True = shared linear velocity, False = independent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = 5000
    epoch = 1
    episode = 0
    train_every_n = 10
    training_iterations = 40
    batch_size = 32
    max_steps = 300
    steps = 0
    save_every = 5
    reward_scale = 0.1  # Scale rewards to prevent Q-value explosion
    
    # ---- Create environment ----
    sim = MARL_2Robot_SIM(
        world_file="robot_nav/worlds/multi_robot_world.yaml",
        disable_plotting=True,  # Set False to visualize
        reward_phase=1,
        active_robot_ids=active_robot_ids,
        coupled_mode=coupled_mode,
    )
    
    # ---- Create model ----
    model = marlTD3_2robot(
        state_dim=state_dim,
        num_total_robots=num_total_robots,
        active_robot_ids=active_robot_ids,
        device=device,
        coupled_mode=coupled_mode,
        save_every=save_every,
        load_model=False,
        model_name="TDR-MARL-2robot-train",
        attention="igs",
        # Load pretrained attention from decentralized model
        load_pretrained_attention=True,
        pretrained_attention_model_name="TDR-MARL-train",
        pretrained_attention_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        freeze_attention=True,  # Freeze attention, only train policy
    )
    
    # ---- Replay buffer ----
    # Buffer stores: (all_robot_states, active_actions, summed_reward, done, all_robot_next_states)
    # State shape: (5 * 11,) = (55,) for all robots
    # Action shape: (4,) for decoupled mode, (3,) for coupled mode
    replay_buffer = ReplayBuffer(buffer_size=50000)
    
    # ---- Initial step ----
    (
        active_poses, active_distances, active_coss, active_sins,
        active_collisions, active_goals, _, active_rewards,
        active_positions, active_goal_positions, all_poses, _
    ) = sim.reset()
    
    running_goals = 0
    running_collisions = 0
    running_reward = 0
    
    # Initialize previous action (scaled action format used in prepare_state)
    prev_action = [[0, 0], [0, 0]] if not coupled_mode else [0, 0, 0]
    
    pbar = tqdm(total=max_epochs, desc="Training")
    
    # ---- Main training loop ----
    while epoch < max_epochs:
        # Prepare state for all robots (for attention network)
        # Use the previous action from the last step
        state = model.prepare_state(
            all_poses, active_distances, active_coss, active_sins,
            active_collisions, prev_action, active_goal_positions
        )
        
        # Get action for 2 active robots
        action, connection, combined_weights = model.get_action(np.array(state), add_noise=True)
        
        # Convert action to environment format
        if coupled_mode:
            a_in = action  # [lin, ang1, ang2]
            a_in_scaled = [(a_in[0] + 1) / 4, a_in[1], a_in[2]]
            prev_action = a_in_scaled  # Save for next state preparation
        else:
            a_in = [[(a[0] + 1) / 4, a[1]] for a in action]
            prev_action = a_in  # Save for next state preparation
        
        # Step environment
        (
            active_poses, active_distances, active_coss, active_sins,
            active_collisions, active_goals, _, active_rewards,
            active_positions, active_goal_positions, all_poses, episode_done
        ) = sim.step(a_in if not coupled_mode else a_in_scaled)
        
        running_goals += sum(active_goals)
        running_collisions += sum(active_collisions)
        running_reward += sum(active_rewards)
        steps += 1
        
        # Prepare next state using the SCALED action (a_in), not raw action
        next_state = model.prepare_state(
            all_poses, active_distances, active_coss, active_sins,
            active_collisions, a_in, active_goal_positions
        )
        
        # Flatten states and add to buffer
        flat_state = np.array(state).flatten()
        flat_next_state = np.array(next_state).flatten()
        flat_action = np.array(action).flatten() if not coupled_mode else np.array(action)
        summed_reward = sum(active_rewards) * reward_scale  # Scale rewards to prevent Q explosion
        
        replay_buffer.add(
            flat_state, flat_action, summed_reward, episode_done, flat_next_state
        )
        
        # Episode termination
        if episode_done or steps >= max_steps or outside_of_bounds(active_poses):
            episode += 1
            steps = 0
            
            # Reset and get initial observations
            (
                active_poses, active_distances, active_coss, active_sins,
                active_collisions, active_goals, _, active_rewards,
                active_positions, active_goal_positions, all_poses, _
            ) = sim.reset()
            
            # Reset previous action for the new episode
            prev_action = [[0, 0], [0, 0]] if not coupled_mode else [0, 0, 0]
            
            # Training
            if episode % train_every_n == 0 and replay_buffer.size() > batch_size * 2:
                model.train(replay_buffer, training_iterations, batch_size)
                epoch += 1
                pbar.update(1)
                
                # Logging
                model.writer.add_scalar("episode/goals", running_goals, epoch)
                model.writer.add_scalar("episode/collisions", running_collisions, epoch)
                model.writer.add_scalar("episode/reward", running_reward, epoch)
                
                running_goals = 0
                running_collisions = 0
                running_reward = 0
    
    pbar.close()
    print("Training complete!")
    model.save(model.model_name, model.save_directory)


if __name__ == "__main__":
    main()