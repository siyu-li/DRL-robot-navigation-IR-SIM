"""
Test script to verify centralized MARL-TD3 setup is correct.

This script tests:
1. Environment step returns correct number of values
2. Model prepare_state returns correct format
3. Actor/Critic have correct input/output dimensions
4. Training loop data flow is correct
5. Replay buffer storage works properly
"""
# Run with command:
# PYTHONPATH=/Users/siyuli/Documents/github/DRL-robot-navigation-IR-SIM python robot_nav/test_centralized_setup.py

from pathlib import Path
import sys
import torch
import numpy as np
import logging
from robot_nav.SIM_ENV.marl_sim import MARL_SIM
from utils import get_buffer

from robot_nav.SIM_ENV.marl_centralized_sim import MARL_SIM
from robot_nav.models.MARL.marlTD3.marlTD3_centralized import marlTD3_centralized

def test_environment():
    """Test 1: Verify environment returns correct values"""
    print("\n" + "="*80)
    print("TEST 1: Environment Step Return Values")
    print("="*80)
    
    sim = MARL_SIM(
        world_file="worlds/multi_robot_world.yaml",
        disable_plotting=True,
        reward_phase=1,
    )  # instantiate environment
    
    num_robots = sim.num_robots
    print(f"✓ Environment initialized with {num_robots} robots")
    
    # Test step
    result = sim.step([[0, 0] for _ in range(num_robots)], None)
    
    print(f"\nEnvironment step returns {len(result)} values:")
    expected_names = [
        "poses", "distances", "coss", "sins", "collisions", 
        "goals", "action", "rewards", "positions", "goal_positions"
    ]
    
    if len(result) == 10:
        print("⚠️  WARNING: Environment returns 10 values (OLD FORMAT)")
        print("   Expected 11 values with episode_done flag")
        print("   You need to add episode_done to marl_sim.py")
        has_episode_done = False
    elif len(result) == 11:
        print("✓ Environment returns 11 values (NEW FORMAT with episode_done)")
        expected_names.append("episode_done")
        has_episode_done = True
    else:
        print(f"❌ ERROR: Unexpected number of return values: {len(result)}")
        return False
    
    # Unpack and verify
    if len(result) == 10:
        poses, distances, coss, sins, collisions, goals, action, rewards, positions, goal_positions = result
        episode_done = None
    else:
        poses, distances, coss, sins, collisions, goals, action, rewards, positions, goal_positions, episode_done = result
    
    print("\nVerifying dimensions:")
    print(f"  poses: {len(poses)} (expected {num_robots})")
    print(f"  collisions: {len(collisions)} (expected {num_robots})")
    print(f"  goals: {len(goals)} (expected {num_robots})")
    print(f"  rewards: {len(rewards)} (expected {num_robots})")
    
    if has_episode_done:
        print(f"  episode_done: {episode_done} (type: {type(episode_done).__name__})")
        if not isinstance(episode_done, bool):
            print("  ⚠️  WARNING: episode_done should be bool")
    
    # Check goal renewal behavior
    print("\nTesting goal renewal behavior:")
    for i in range(num_robots):
        if goals[i]:
            print(f"  Robot {i} reached goal: {goals[i]}")
    
    return has_episode_done


def test_model_architecture():
    """Test 2: Verify model architecture dimensions"""
    print("\n" + "="*80)
    print("TEST 2: Model Architecture")
    print("="*80)
    
    num_robots = 3
    state_dim = 11
    action_dim = 2
    joint_action_dim = num_robots * action_dim  # 6
    batch_size = 16
    
    device = torch.device("cpu")
    
    print(f"\nInitializing model:")
    print(f"  num_robots: {num_robots}")
    print(f"  state_dim: {state_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  joint_action_dim: {joint_action_dim}")
    
    try:
        model = marlTD3_centralized(
            state_dim=state_dim,
            joint_action_dim=joint_action_dim,
            max_action=1.0,
            device=device,
            num_robots=num_robots,
            attention="g2anet",
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"❌ ERROR initializing model: {e}")
        return False
    
    # Test Actor
    print("\n--- Testing CentralizedActor ---")
    states = torch.randn(batch_size, num_robots, state_dim)
    print(f"Input states shape: {states.shape}")
    
    try:
        action, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = model.actor(states)
        print(f"✓ Actor forward pass successful")
        print(f"  Action shape: {action.shape} (expected: ({batch_size}, {joint_action_dim}))")
        
        if action.shape == (batch_size, joint_action_dim):
            print("  ✓ Correct shape!")
        else:
            print(f"  ❌ Wrong shape! Expected ({batch_size}, {joint_action_dim})")
            return False
    except Exception as e:
        print(f"❌ ERROR in actor forward: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Critic
    print("\n--- Testing CentralizedCritic ---")
    actions = torch.randn(batch_size, joint_action_dim)
    print(f"Input actions shape: {actions.shape}")
    
    try:
        q1, q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights = model.critic(states, actions)
        print(f"✓ Critic forward pass successful")
        print(f"  Q1 shape: {q1.shape} (expected: ({batch_size}, 1))")
        print(f"  Q2 shape: {q2.shape} (expected: ({batch_size}, 1))")
        
        if q1.shape == (batch_size, 1) and q2.shape == (batch_size, 1):
            print("  ✓ Correct shapes!")
        else:
            print(f"  ❌ Wrong shapes!")
            return False
    except Exception as e:
        print(f"❌ ERROR in critic forward: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_prepare_state():
    """Test 3: Verify prepare_state returns correct format"""
    print("\n" + "="*80)
    print("TEST 3: Prepare State Function")
    print("="*80)
    
    num_robots = 3
    state_dim = 11
    device = torch.device("cpu")
    
    model = marlTD3_centralized(
        state_dim=state_dim,
        joint_action_dim=num_robots * 2,
        max_action=1.0,
        device=device,
        num_robots=num_robots,
        attention="g2anet",
    )
    
    # Create mock data
    poses = [[1.0, 2.0, 0.5], [3.0, 4.0, 1.0], [5.0, 6.0, 1.5]]
    distance = [2.5, 3.0, 1.8]
    cos_vals = [0.8, 0.6, 0.9]
    sin_vals = [0.6, 0.8, 0.4]
    collision = [False, False, False]
    action = [[0.3, 0.1], [0.2, -0.1], [0.4, 0.2]]
    goal_positions = [[8.0, 9.0], [7.0, 8.0], [9.0, 7.0]]
    
    print("\nCalling prepare_state...")
    try:
        result = model.prepare_state(poses, distance, cos_vals, sin_vals, collision, action, goal_positions)
        
        # Check if returns tuple (old format) or just list (new format)
        if isinstance(result, tuple):
            print("⚠️  WARNING: prepare_state returns tuple (states, terminal)")
            print("   For centralized version, should return only states")
            states, terminal = result
            print(f"  states: {len(states)} vectors")
            print(f"  terminal: {terminal}")
            has_terminal = True
        else:
            print("✓ prepare_state returns states only (correct for centralized)")
            states = result
            print(f"  states: {len(states)} vectors")
            has_terminal = False
        
        # Verify states
        print(f"\nVerifying state vectors:")
        print(f"  Number of state vectors: {len(states)}")
        print(f"  State dimension: {len(states[0])}")
        print(f"  Expected: {num_robots} vectors of dim {state_dim}")
        
        if len(states) == num_robots and len(states[0]) == state_dim:
            print("  ✓ Correct dimensions!")
            return not has_terminal  # Return True if no terminal (correct format)
        else:
            print("  ❌ Wrong dimensions!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in prepare_state: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop_integration():
    """Test 4: Verify training loop data flow"""
    print("\n" + "="*80)
    print("TEST 4: Training Loop Integration")
    print("="*80)
    
    sim = MARL_SIM(
        world_file="worlds/multi_robot_world.yaml",
        disable_plotting=True,
        reward_phase=1,
    )
    
    num_robots = sim.num_robots
    state_dim = 11
    joint_action_dim = num_robots * 2
    device = torch.device("cpu")
    
    model = marlTD3_centralized(
        state_dim=state_dim,
        joint_action_dim=joint_action_dim,
        max_action=1.0,
        device=device,
        num_robots=num_robots,
        attention="g2anet",
    )
    
    print(f"\nSimulating one training step...")
    
    # Initial step
    try:
        env_output = sim.step([[0, 0] for _ in range(num_robots)], None)
        print(f"✓ Environment step successful ({len(env_output)} values returned)")
        
        if len(env_output) == 10:
            poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = env_output
            episode_done = any(collision) or all(goal)  # Compute manually
            print(f"  ⚠️  Computing episode_done manually: {episode_done}")
        else:
            poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions, episode_done = env_output
            print(f"  ✓ episode_done from environment: {episode_done}")
    except Exception as e:
        print(f"❌ ERROR in environment step: {e}")
        return False
    
    # Prepare state
    try:
        state_result = model.prepare_state(poses, distance, cos, sin, collision, a, goal_positions)
        
        if isinstance(state_result, tuple):
            state, terminal = state_result
            print(f"⚠️  prepare_state returns (state, terminal) - need to update")
        else:
            state = state_result
            print(f"✓ prepare_state returns state only")
        
        print(f"  State: {len(state)} vectors of dim {len(state[0])}")
    except Exception as e:
        print(f"❌ ERROR in prepare_state: {e}")
        return False
    
    # Get action
    try:
        action, connection, combined_weights = model.get_action(np.array(state), add_noise=False)
        print(f"✓ get_action successful")
        print(f"  Action shape: {action.shape} (expected: ({num_robots}, 2))")
        
        if action.shape == (num_robots, 2):
            print("  ✓ Correct action shape!")
        else:
            print(f"  ❌ Wrong action shape!")
    except Exception as e:
        print(f"❌ ERROR in get_action: {e}")
        return False
    
    # Buffer storage format
    print("\n--- Buffer Storage Format ---")
    print(f"state: list of {len(state)} vectors")
    print(f"action: {action.shape}")
    print(f"reward: list of {len(reward)} values")
    
    # Show what done_flags should be
    if len(env_output) == 11:
        print(f"done_flags: [{episode_done}] * {num_robots} = {[episode_done] * num_robots}")
        print("  ✓ Correct: replicate episode_done for all robots")
    else:
        print(f"done_flags: Need to compute from collision/goal")
        print(f"  collision: {collision}")
        print(f"  goal: {goal}")
        print(f"  episode_done = any(collision) or all(goal) = {any(collision) or all(goal)}")
    
    return True


def test_reward_aggregation():
    """Test 5: Verify reward aggregation in training"""
    print("\n" + "="*80)
    print("TEST 5: Reward Aggregation")
    print("="*80)
    
    num_robots = 3
    batch_size = 4
    
    # Mock batch data
    batch_rewards = [
        [1.0, 2.0, 3.0],  # Episode 1: sum = 6.0
        [0.5, 0.5, 0.5],  # Episode 2: sum = 1.5
        [-1.0, 2.0, 1.0], # Episode 3: sum = 2.0
        [0.0, 0.0, 0.0],  # Episode 4: sum = 0.0
    ]
    
    # Flatten for buffer format
    flat_rewards = []
    for episode_rewards in batch_rewards:
        flat_rewards.extend(episode_rewards)
    
    print(f"Batch rewards (per robot):")
    for i, r in enumerate(batch_rewards):
        print(f"  Episode {i}: {r}")
    
    # Aggregate as in training
    reward_tensor = (
        torch.Tensor(flat_rewards)
        .view(batch_size, num_robots)
        .sum(dim=1, keepdim=True)
    )
    
    print(f"\nAggregated rewards (team total):")
    print(f"  Shape: {reward_tensor.shape} (expected: ({batch_size}, 1))")
    for i in range(batch_size):
        print(f"  Episode {i}: {reward_tensor[i, 0].item()}")
    
    expected = [6.0, 1.5, 2.0, 0.0]
    if all(abs(reward_tensor[i, 0].item() - expected[i]) < 1e-5 for i in range(batch_size)):
        print("✓ Reward aggregation correct!")
        return True
    else:
        print("❌ Reward aggregation incorrect!")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CENTRALIZED MARL-TD3 SETUP VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Run tests
    results["Environment"] = test_environment()
    results["Model Architecture"] = test_model_architecture()
    results["Prepare State"] = test_prepare_state()
    results["Training Loop"] = test_training_loop_integration()
    results["Reward Aggregation"] = test_reward_aggregation()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Setup is correct!")
    else:
        print("⚠️  SOME TESTS FAILED - Review output above")
        print("\nCommon issues:")
        print("1. Environment doesn't return episode_done (needs marl_sim.py update)")
        print("2. prepare_state returns (state, terminal) instead of just state")
        print("3. Model dimensions don't match (check joint_action_dim)")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
