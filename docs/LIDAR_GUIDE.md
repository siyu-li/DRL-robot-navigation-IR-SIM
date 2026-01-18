# MARL Navigation with LiDAR and Obstacle Avoidance

This document describes how to use the LiDAR-based multi-agent navigation features.

## Features

- **LiDAR observations**: Each robot gets a 2D LiDAR scan (normalized to `[0, 1]`) as part of its observation
- **Obstacle randomization**: Obstacles can be randomly placed each episode
- **Late fusion architecture**: LiDAR is fused AFTER attention (attention network remains unchanged)
- **Augmented state**: LiDAR is concatenated with agent state, reuses original ReplayBuffer
- **Multiple encoder options**: Sector (recommended), MLP, or CNN encoders available
- **Backward compatible**: LiDAR features are optional; existing code works unchanged

## Quick Start

### 1. Train with LiDAR and Random Obstacles

```bash
# Full training with visualization disabled (recommended for speed)
python -m robot_nav.marl_train_lidar

# The default settings are:
#   use_lidar=True
#   random_obstacles=True
#   lidar_num_beams=180
#   lidar_embed_dim=64
```

### 2. Train without LiDAR (Original Behavior)

```bash
# Uses original MARL training (no LiDAR)
python -m robot_nav.marl_train
```

## Configuration Options

### Key Hyperparameters in `marl_train_lidar.py`

| Parameter | Default | Description |
|----------|---------|-------------|
| `use_lidar` | True | Enable LiDAR observations |
| `lidar_num_beams` | 180 | Number of LiDAR beams |
| `lidar_range_max` | 7.0 | Maximum LiDAR range (meters) |
| `lidar_embed_dim` | 64 | LiDAR embedding dimension |
| `random_obstacles` | True | Randomize obstacles each episode |
| `num_obstacles` | 5 | Number of obstacles to randomize |
| `max_epochs` | 50 | Maximum training epochs |
| `batch_size` | 16 | Training batch size |
| `max_steps` | 300 | Maximum steps per episode |

### YAML World Configuration

The LiDAR sensor is configured in the world YAML file. See `robot_nav/worlds/multi_robot_world_lidar.yaml`:

```yaml
robot:
  - number: 5
    kinematics: {name: 'diff'}
    # ... other config ...

    sensors:
      - type: 'lidar2d'
        range_min: 0
        range_max: 7        # Maximum sensing range
        angle_range: 3.14   # 180 degrees (PI radians)
        number: 180         # Number of beams
        noise: True         # Add sensor noise
        std: 0.08           # Noise standard deviation
        angle_std: 0.1      # Angular noise std
        offset: [0, 0, 0]   # Sensor offset from robot center

obstacle:
  - number: 3
    shape: {name: 'circle', radius: 0.5}
    distribution: {name: 'random', range_low: [1, 1, -3.14], range_high: [11, 11, 3.14]}
    kinematics: {name: 'static'}
```

## Architecture Overview

### Key Design Principles

1. **Attention network is UNCHANGED**: Original IGA/G2ANet attention processes only agent state (11-dim)
2. **Late fusion**: LiDAR is encoded separately and fused AFTER attention in the policy head
3. **Augmented state**: `[agent_state (11-dim) | lidar (num_beams)]` stored in original ReplayBuffer
4. **LiDAR values in `[0, 1]`**: Normalized by `lidar_range_max` in the simulation environment

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Augmented State                                    │
│    [agent_state (11-dim) | lidar_scan (180 beams)] = 191-dim total          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       _split_state()          │
                    │  Splits into agent + lidar   │
                    └──────────────────────────────┘
                           /              \
                          /                \
                         ▼                  ▼
        ┌────────────────────────┐  ┌────────────────────────┐
        │   Agent State (11-dim) │  │  LiDAR Scan (180-dim)  │
        │   [px, py, cos, sin,   │  │  Normalized to [0, 1]  │
        │    dist, cos_g, sin_g, │  │  (ranges / range_max)  │
        │    lin_vel, ang_vel,   │  │                        │
        │    gx, gy]             │  │                        │
        └────────────────────────┘  └────────────────────────┘
                    │                          │
                    ▼                          ▼
        ┌────────────────────────┐  ┌────────────────────────┐
        │   ORIGINAL Attention   │  │    LiDAR Encoder       │
        │   (IGA or G2ANet)      │  │  (Sector/MLP/CNN)      │
        │   *** UNCHANGED ***    │  │                        │
        └────────────────────────┘  └────────────────────────┘
                    │                          │
                    ▼                          ▼
        ┌────────────────────────┐  ┌────────────────────────┐
        │  attn_out (512-dim)    │  │ lidar_embed (N-dim)    │
        └────────────────────────┘  └────────────────────────┘
                    \                        /
                     \                      /
                      ▼                    ▼
              ┌────────────────────────────────┐
              │         Late Fusion            │
              │   torch.cat([attn_out,         │
              │              lidar_embed])     │
              │   → (512 + lidar_embed_dim)    │
              └────────────────────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │         Policy Head            │
              │   Linear → LeakyReLU           │
              │   Linear → LeakyReLU           │
              │   Linear → Tanh                │
              └────────────────────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │      Actions (2-dim)           │
              │   [linear_vel, angular_vel]    │
              └────────────────────────────────┘
```

### Why Late Fusion?

LiDAR doesn't meaningfully influence robot-to-robot attention because:
- Attention operates on **edge features** (relative geometry between robots)
- LiDAR sees **obstacles** (walls, boxes), not other robots
- Fusing early would add complexity without clear benefit

Instead:
- ✅ **Attention** handles "which robots should I coordinate with?"
- ✅ **LiDAR** handles "what obstacles are around me?"
- ✅ **Policy head** combines both to make final decisions

### LiDAR Encoder Options

Three encoder types are available (defined in `robot_nav/models/MARL/lidar_encoder.py`):

#### Option 1: Sector Encoder (Recommended - Simplest)

Divides the 180° scan into sectors and takes the minimum distance per sector:

```python
from robot_nav.models.MARL.lidar_encoder import LiDARSectorEncoder

# 12 sectors = 15° each, output is 12-dim
encoder = LiDARSectorEncoder(
    num_beams=180, 
    num_sectors=12, 
    aggregation="min"  # or "mean", "both"
)

# Input: normalized LiDAR scan in [0, 1]
lidar_input = torch.rand(5, 180)  # 5 robots, 180 beams, values in [0, 1]
embedding = encoder(lidar_input)  # Shape: (5, 12)
```

**Pros**: Fast, interpretable, few parameters  
**Output**: 12-dim vector (min distance per 15° sector)

#### Option 2: MLP Encoder

Simple feed-forward network:

```python
from robot_nav.models.MARL.lidar_encoder import LiDAREncoderMLP

encoder = LiDAREncoderMLP(
    num_beams=180, 
    output_dim=32, 
    hidden_dim=128, 
    num_layers=2
)
```

**Pros**: Easy to train, good baseline  
**Output**: 32-dim learned embedding

#### Option 3: CNN Encoder

1D convolutions to capture spatial patterns:

```python
from robot_nav.models.MARL.lidar_encoder import LiDAREncoderCNN

encoder = LiDAREncoderCNN(
    num_beams=180, 
    output_dim=32, 
    channels=[16, 32, 32]
)
```

**Pros**: Captures spatial patterns (walls, corners)  
**Output**: 32-dim learned embedding

#### Factory Function

```python
from robot_nav.models.MARL.lidar_encoder import create_lidar_encoder

# Create any encoder type
encoder = create_lidar_encoder(
    encoder_type="sector",  # or "mlp", "cnn"
    num_beams=180,
    output_dim=12,
    num_sectors=12,       # sector-specific
    aggregation="min",    # "min", "mean", or "both"
)
```

**Note**: All encoders expect input normalized to `[0, 1]` range.

## File Structure

```
robot_nav/
├── SIM_ENV/
│   ├── marl_lidar_sim.py      # MARL env with LiDAR (returns normalized [0,1] scans)
│   └── ...
├── models/
│   └── MARL/
│       ├── lidar_encoder.py    # LiDAR encoder modules (Sector/MLP/CNN)
│       └── marlTD3/
│           ├── marlTD3_lidar.py # TD3WithLiDAR (late fusion, augmented state)
│           └── ...
├── worlds/
│   ├── multi_robot_world_lidar.yaml  # World with LiDAR sensor config
│   └── ...
├── replay_buffer.py            # Original buffer (reused with augmented state)
├── marl_train_lidar.py         # Training script with LiDAR
└── ...
```

## State Format and ReplayBuffer Compatibility

### Augmented State Design

The key design choice is to concatenate LiDAR with agent state to form an **augmented state**:

```
Augmented State = [agent_state (11-dim) | lidar_scan (num_beams)]
                = [px, py, cos_h, sin_h, dist/17, cos_g, sin_g, lin_v, ang_v, gx, gy | lidar...]
```

Total dimension: `11 + lidar_num_beams` (e.g., `11 + 180 = 191`)

### Why Augmented State?

1. **Reuses original ReplayBuffer**: No modifications needed
2. **Simple state management**: Single array per robot
3. **Consistent with existing training loop**: Just change state dimension

### LiDAR Normalization

LiDAR readings are normalized to `[0, 1]` in the simulation environment:

```python
# In MARL_LIDAR_SIM._get_single_lidar_scan()
scan_data = self.env.get_lidar_scan(robot_id)
return scan_data["ranges"] / self.lidar_range_max  # Normalized to [0, 1]
```

This means:
- `0.0` = obstacle at sensor (very close)
- `1.0` = obstacle at max range (or no obstacle)

The LiDAR encoder expects this `[0, 1]` normalized input.

## API Reference

### MARL_LIDAR_SIM

```python
from robot_nav.SIM_ENV.marl_lidar_sim import MARL_LIDAR_SIM

sim = MARL_LIDAR_SIM(
    world_file="robot_nav/worlds/multi_robot_world_lidar.yaml",
    disable_plotting=True,
    reward_phase=1,
    use_lidar=True,
    lidar_num_beams=180,
    lidar_range_max=7.0,
    random_obstacles=True,
    num_obstacles=5,
)

# Reset returns 11 values when use_lidar=True
(poses, distance, cos, sin, collision, goal, action, 
 reward, positions, goal_positions, lidar_scans) = sim.reset()

# lidar_scans: List of normalized [0, 1] arrays, one per robot
# Each array has shape (num_beams,)

# Step also returns lidar_scans as last element
result = sim.step(actions, connections)
lidar_scans = result[-1]  # List of normalized scans per robot
```

### TD3WithLiDAR

```python
from robot_nav.models.MARL.marlTD3.marlTD3_lidar import TD3WithLiDAR

model = TD3WithLiDAR(
    state_dim=11,               # Agent state dim (WITHOUT LiDAR)
    action_dim=2,
    max_action=1.0,
    device=torch.device("cuda"),
    num_robots=5,
    attention="igs",            # or "g2anet"
    # LiDAR config
    use_lidar=True,
    lidar_encoder_type="sector",  # "sector", "mlp", or "cnn"
    lidar_num_beams=180,
    lidar_embed_dim=12,         # Output dim of encoder
    lidar_encoder_kwargs={"num_sectors": 12, "aggregation": "min"},
    lidar_range_max=7.0,
)
```

### prepare_state Method

The `prepare_state` method creates augmented states compatible with the original ReplayBuffer:

```python
# Returns augmented states and terminal flags
state, terminal = model.prepare_state(
    poses,           # List of [x, y, theta] per robot
    distance,        # List of distances to goal
    cos,             # List of cos(heading error)
    sin,             # List of sin(heading error)
    collision,       # List of collision flags
    action,          # List of last actions
    goal_positions,  # List of [gx, gy] per robot
    lidar_scans,     # List of normalized [0, 1] LiDAR arrays (optional)
)

# state: List of augmented states, each is (11 + num_beams,) if use_lidar
# terminal: List of collision flags
```

**Internal state construction**:

```python
# Agent state (11-dim):
agent_state = [
    px, py,           # Position
    cos(theta),       # Heading cosine
    sin(theta),       # Heading sine
    distance / 17,    # Normalized distance to goal
    cos_g, sin_g,     # Angle to goal
    lin_vel * 2,      # Scaled linear velocity
    (ang_vel + 1) / 2, # Scaled angular velocity
    gx, gy,           # Goal position
]

# If use_lidar: concatenate lidar scan (already in [0, 1])
augmented = np.concatenate([agent_state, lidar_scan])
```

### get_action Method

```python
# Get action from augmented state
action, connection, combined_weights = model.get_action(
    np.array(state),  # Augmented states shape: (N, 11 + num_beams)
    add_noise=True,   # Add exploration noise
)

# action: shape (N, 2) - [linear_vel, angular_vel] per robot
# connection: attention connection logits
# combined_weights: attention weights for visualization
```

### Using with Original ReplayBuffer

```python
from robot_nav.replay_buffer import ReplayBuffer

replay_buffer = ReplayBuffer(buffer_size=10000)

# Add experience with augmented state
replay_buffer.add(
    state,      # Augmented state: (N, 11 + num_beams)
    action,     # Actions: (N, 2)
    reward,     # Rewards: (N,)
    terminal,   # Done flags: (N,)
    next_state, # Next augmented state: (N, 11 + num_beams)
)

# Training uses augmented states directly
model.train(
    replay_buffer=replay_buffer,
    iterations=80,
    batch_size=16,
)
```

### LiDAREncoder

```python
from robot_nav.models.MARL.lidar_encoder import (
    create_lidar_encoder,
    LiDARSectorEncoder,
    LiDAREncoderMLP,
    LiDAREncoderCNN,
)

# Option 1: Sector encoder (recommended)
encoder = create_lidar_encoder("sector", num_beams=180, output_dim=12, num_sectors=12)

# Option 2: MLP encoder
encoder = create_lidar_encoder("mlp", num_beams=180, output_dim=32)

# Option 3: CNN encoder
encoder = create_lidar_encoder("cnn", num_beams=180, output_dim=32)

# Usage - input must be normalized to [0, 1]
lidar_input = torch.rand(5, 180)  # 5 robots, 180 beams, values in [0, 1]
embedding = encoder(lidar_input)  # Shape depends on encoder
```

## Training Loop Example

Here's a complete training loop showing how LiDAR integrates with the existing system:

```python
from robot_nav.models.MARL.marlTD3.marlTD3_lidar import TD3WithLiDAR
from robot_nav.SIM_ENV.marl_lidar_sim import MARL_LIDAR_SIM
from robot_nav.replay_buffer import ReplayBuffer

# Initialize
sim = MARL_LIDAR_SIM(
    use_lidar=True,
    lidar_num_beams=180,
    lidar_range_max=7.0,
)

model = TD3WithLiDAR(
    state_dim=11,
    action_dim=2,
    max_action=1.0,
    num_robots=sim.num_robots,
    device=torch.device("cuda"),
    use_lidar=True,
    lidar_num_beams=180,
    lidar_embed_dim=64,
)

replay_buffer = ReplayBuffer(buffer_size=50000)

# Initial step
connections = torch.zeros(sim.num_robots, sim.num_robots - 1)
result = sim.step([[0, 0] for _ in range(sim.num_robots)], connections)
poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions, lidar_scans = result

# Training loop
for episode in range(num_episodes):
    # Prepare augmented state
    state, terminal = model.prepare_state(
        poses, distance, cos, sin, collision, a, goal_positions, lidar_scans
    )
    
    # Get action from augmented state
    action, connection, combined_weights = model.get_action(np.array(state), add_noise=True)
    
    # Step environment
    result = sim.step(action, connection, combined_weights)
    poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions, lidar_scans = result
    
    # Prepare next augmented state
    next_state, terminal = model.prepare_state(
        poses, distance, cos, sin, collision, a, goal_positions, lidar_scans
    )
    
    # Add to buffer (augmented states stored directly)
    replay_buffer.add(state, action, reward, terminal, next_state)
    
    # Train
    if episode % train_every_n == 0:
        model.train(replay_buffer=replay_buffer, iterations=80, batch_size=16)
```

## Troubleshooting

### LiDAR scans are all 1.0 (max range)

- Check that obstacles are within range in your world file
- Verify the robot sensors are configured correctly
- Try reducing `lidar_range_max` to see if readings appear

### Dimension mismatch errors

- Ensure `lidar_num_beams` matches across `MARL_LIDAR_SIM`, `TD3WithLiDAR`, and world YAML
- Check that `state_dim=11` (without LiDAR) is passed to `TD3WithLiDAR`
- Verify augmented state is `(11 + num_beams,)` per robot

### Performance is slow

- Use `disable_plotting=True` for headless training
- Use sector encoder (fastest) instead of CNN
- Reduce `lidar_num_beams` if using many robots

### Encoder expects different input range

- All encoders expect input in `[0, 1]` range
- The simulation already normalizes: `ranges / lidar_range_max`
- If using custom data, ensure normalization is applied
