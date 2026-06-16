# CAPSwitcher: Coarse-And-Precise Switcher for Coupled Unicycle Swarms

---

## 1. Project Background

### The Physical System

We consider a swarm of coupled unicycle robots organized into overlapping groups. Each group defines a collective actuation mode. When a group is activated, **all robots in the swarm rotate** — not just the group members. The rotation amount for each robot is determined by the actuation matrix **A**. After rotation, all robots translate forward by the same distance, each along its own heading.

The group structure is defined by a binary **membership matrix** **M**, where M[i, s] = 1 means robot i belongs to group s. For example, with 6 robots and 4 groups:

| | Robot 1 | Robot 2 | Robot 3 | Robot 4 | Robot 5 | Robot 6 |
|---|---|---|---|---|---|---|
| Group 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| Group 2 | 0 | 1 | 1 | 0 | 0 | 1 |
| Group 3 | 1 | 0 | 1 | 0 | 1 | 0 |
| Group 4 | 0 | 0 | 0 | 0 | 0 | 0 |

The **actuation matrix A** is derived from the membership matrix by **complementing** each entry: 1 → 0, 0 → 1. This means robots that do **not** belong to a group are the ones that rotate when that group is activated:

| | Group 1 | Group 2 | Group 3 | Group 4 |
|---|---|---|---|---|
| Robot 1 | **1** | **1** | 0 | **1** |
| Robot 2 | **1** | 0 | **1** | **1** |
| Robot 3 | **1** | 0 | 0 | **1** |
| Robot 4 | 0 | **1** | **1** | **1** |
| Robot 5 | 0 | **1** | 0 | **1** |
| Robot 6 | 0 | 0 | **1** | **1** |

Group 4 (membership all-zeros) complements to all-ones in A, providing a mode where **all robots rotate by the same angle**.

When group s is activated with rotation parameter t, each robot i rotates by dθ\_i = A[i, s] · t. Then all robots drive forward along their individual headings by the same distance d. The total rotation from activating multiple groups with parameters **t** = [t1, t2, ..., t_K] is **dθ = A · t**.

### The Rank-Deficiency Problem

The actuation matrix **A** is rank-deficient: it has fewer independent columns than robots. This means the system cannot independently control all robot headings simultaneously. Only a low-dimensional subspace of the full rotation space is reachable through any combination of group activations. For the simplified 6-robot system, we artificially constrain to rank 2 (by using only 2 of 3 groups) to study this limitation.

### Two Control Regimes

**Coarse group control.** When a coarse group is activated, all robots rotate according to the actuation matrix (robots outside the group membership rotate by t, group members do not), then all robots drive forward by the same distance along their individual headings. This is physically efficient — every robot makes forward progress with no reversal or back-and-forth. However, due to rank deficiency of A, the rotation cannot independently steer each robot. The swarm can only adjust headings within the low-dimensional controllable subspace. Obstacle avoidance is not possible at this level.

**Precise (dense) group control.** Individual robots can be steered independently by composing multiple group activations. For example, to move a specific subset of 3 robots in a desired direction, the system activates a coarse group (all robots rotate and move forward), then activates additional groups to rotate certain robots by 180° and move them backward, effectively reversing the unwanted motion for robots outside the target subset. This is achieved through the existing Graph Attention Network (GAT) policy, which provides per-robot navigation actions with obstacle avoidance. However, precise control is physically wasteful: the back-and-forth reversal means robots travel extra distance to achieve the net effect of moving only a subset.

### The Efficiency Trade-Off

- **Coarse control:** high physical efficiency (all robots advance), low precision (rank-deficient steering, no obstacle avoidance).
- **Precise control:** high precision (individual steering, obstacle-aware), low physical efficiency (back-and-forth reversal wastes motion).

A naive policy that always uses precise control reaches the goal but wastes physical motion. A naive policy that always uses coarse control is efficient but collides with obstacles and cannot navigate precisely. A previous learned switcher based on minimum-action group policies always selected precise control, because the coarse group policy (using minimum action across individual robot policies) produced near-zero motion — the switcher correctly learned that coarse control was useless.

---

## 2. Task Description

### The Core Problem

Design a learned switcher — the **CAPSwitcher** (Coarse-And-Precise Switcher) — that decides at each timestep whether to use coarse group control or precise individual control. The switcher should learn to:

- Use **coarse control in open space**, where obstacles are far away and the swarm needs to cover distance efficiently.
- Use **precise control near obstacles and near the goal**, where individual steering and obstacle avoidance are critical.

### Enabling the Coarse Group

The key prerequisite (and a contribution of this work) is a viable coarse group steering mechanism. We replace the failed minimum-action approach with **least-squares steering** (or nonlinear progress optimization) over the rank-deficient actuation matrix. Given desired robot headings toward the goal, we solve:

```
t* = pinv(A_reduced) · dθ_desired       (least-squares heading)
```

or

```
t*, d* = argmax total_progress(x, t, d)  (nonlinear progress optimization)
```

This produces meaningful collective motion despite rank deficiency, giving the switcher a coarse control option worth selecting.

---

## 3. Project Goals

### Primary Goal
Demonstrate that a learned CAPSwitcher, choosing between least-squares coarse steering and a GAT-based precise policy, produces more physically efficient multi-robot navigation than either control mode alone.

### Specific Objectives

1. **Validate coarse steering.** Show that least-squares (or nonlinear-optimized) steering over a rank-deficient actuation matrix produces meaningful forward progress in free space. Compare least-squares heading optimization vs. nonlinear progress optimization.

2. **Train the CAPSwitcher.** Train an RL-based switcher that takes state information (via frozen GAT embeddings with a learned projection) and outputs a binary decision: coarse or precise. The reward must capture both goal-reaching and physical efficiency.

3. **Demonstrate emergent coarse-to-fine behavior.** Show that the switcher learns to use coarse control in open space and transitions to precise control near obstacles and the goal, without this behavior being hard-coded.

4. **Quantify efficiency gains.** Compare against baselines on total physical distance traveled, success rate, and planning time.

### Baselines
- **Baseline 1 — Precise only:** Always use the GAT individual policy. Expected: high success rate, high physical waste.
- **Baseline 2 — Coarse only:** Always use least-squares coarse steering. Expected: efficient in open space, fails near obstacles.
- **Baseline 3 — Heuristic switcher:** Clearance-threshold switching (coarse if clearance > threshold, else precise). Expected: reasonable performance, but rigid boundary.

### Evaluation Metrics
- **Success rate:** fraction of episodes where all robots reach their goals.
- **Physical efficiency:** total distance traveled by all robots / sum of straight-line distances to goals.
- **Switching profile:** visualization of when/where the switcher selects coarse vs. precise control.

---

## 4. GAT Backbone Analysis & CAPSwitcher Architecture

### 4.1 GAT Backbone Layer-by-Layer Summary

The backbone is `AttentionObstacleOptimized` in `robot_nav/models/MARL/Attention/iga_obstacle_optimized.py`, used inside `ActorObstacle` in `marlTD3_obstacle.py`. Its forward pass has four functional stages:

```
Input
  robot_embedding : (B, N, 11)   [px, py, cos θ, sin θ, dist_goal, cos_err, sin_err, lin_vel, ang_vel, gx, gy]
  obstacle_embedding : (B, M, 4) [ox, oy, cos θ_obs, sin θ_obs]

Stage 1 — Node Encoder                      (FREEZE)
  robot_feat = robot_embedding[:, :, 4:9]   5-dim per-robot features
  embedding1 : Linear(5 → 128) + LeakyReLU
  embedding2 : Linear(128 → 256) + LeakyReLU
  → robot_embed : (B, N, 256)

  obs_embed : zeros(B, M, 256)              # obstacle info enters only via edge features

Stage 2 — Hard Attention                    (FREEZE)
  Robot-robot:
    hard_mlp    : Linear(256+7 → 256) → ReLU → Linear(256 → 256)
    hard_encoding: Linear(256 → 2)
    Gumbel-softmax(τ=0.2) → hard_weights_rr : (B, N, N)  binary proximity gate
  Robot-obstacle:
    hard_mlp_obs  : Linear(256+5 → 256) → ReLU → Linear(256 → 256)
    hard_encoding_obs: Linear(256 → 2)
    Gumbel-softmax(τ=0.2) → hard_weights_ro : (B, N, M)  binary proximity gate

Stage 3 — Soft Message Passing              (FREEZE)
  GoalAttentionLayerObstacle (MessagePassing, aggr=add):
    q : Linear(256 → 256, no bias)  — target robot query
    k : Linear(10 → 256)            — edge key
    v : Linear(10 → 256)            — edge value
    attn_score_layer: Linear(512→256)→ReLU→Linear(256→1)
  Hard masks gate which edges enter message passing.
  Soft softmax weights computed per-target-node.
  → attn_out : (B·N, 256)

  ── CUT POINT ──────────────────────────────────────────
  self_embed = robot_embed.reshape(B·N, 256)
  _pre_decoder_embedding = cat(self_embed, attn_out)  → (B·N, 512)
  Stored as self._pre_decoder_embedding (detached) after every forward pass.
  ── CUT POINT ──────────────────────────────────────────

Stage 4 — Decoder / Task Head               (DISCARD for switching)
  decode_1 : Linear(512 → 512) + LeakyReLU
  decode_2 : Linear(512 → 512) + LeakyReLU
  → att_embedding : (B·N, 512)   (navigation-optimised)

Policy Head (in ActorObstacle)              (DISCARD for switching)
  3-layer MLP: 512→400→300→action_dim + Tanh
```

### 4.2 Why `_pre_decoder_embedding` Is the Right Cut Point

| Component | Encodes | Useful for switching? | Decision |
|---|---|---|---|
| Node encoder (emb1/2) | Generic per-robot local state | Yes — same input space | **Freeze** |
| Hard attention (hard_mlp / hard_mlp_obs) | Which neighbors/obstacles are in proximity | **Directly yes** — captures "am I near an obstacle?" | **Freeze** |
| Soft message passing (q/k/v, attn_score) | Relational importance of neighbors | Yes — spatial context | **Freeze** |
| `_pre_decoder_embedding` (512-dim) | Self state ⊕ aggregated neighborhood | **Ideal switching signal** | **Freeze & use as input** |
| Decoder (decode_1/2) | Fuses features optimized for *action prediction* | Navigation-specific, not needed | **Discard** |
| Policy head | Maps to lin/ang velocity | Navigation-specific | **Discard** |

The 512-dim `_pre_decoder_embedding = cat(self_embed[256], attn_out[256])` is **task-agnostic**: it encodes each robot's local state and how spatially constrained it is by neighbors and obstacles — exactly the signal needed to decide coarse vs. precise. The decoder layers on top encode "what action to take for navigation", which is not what the switcher needs.

### 4.3 Updated CAPSwitcher Architecture

```
                        GAT Backbone (ALL FROZEN)
                        ┌─────────────────────────────────────────────────┐
robot_obs (B,N,11) ────▶│  Node Encoder → Hard Attention → Soft Message  │
obstacle_obs (B,M,4)    │  Passing → cat(self_embed, attn_out)            │
                        │                      ↓                          │
                        │     _pre_decoder_embedding  (B·N, 512)          │
                        └───────────────────┬─────────────────────────────┘
                                            │ detach + reshape
                                            ▼
                                     (B, N, 512)
                                            │
                                     Mean pool over N
                                            ▼
                                      (B, 512)       Switcher Head (TRAINED)
                                            │  ┌──────────────────────────┐
                                            └─▶│  MLP: 512→256→128→2      │
                                               │  → logits {coarse,precise}│
                                               │  Categorical distribution  │
                                               └──────────────────────────┘
                                                          │
                        ┌─────────────────┐               │  ┌──────────────────────┐
                        │  Coarse Steering│◀── action=0 ──┘  │  Precise (frozen GAT)│
                        │  pinv(A)·dθ     │                  │  actor_target forward │
                        └─────────────────┘    action=1 ────▶└──────────────────────┘
```

### 4.4 Training Setup (Revised)

| Module | Weights | Updated during switcher training? |
|---|---|---|
| `AttentionObstacleOptimized` (all 4 stages) | Pre-trained TD3Obstacle actor | **No — fully frozen** |
| `ActorObstacle.policy_head` | Pre-trained TD3Obstacle | **No — used only for precise actions** |
| Switcher Head MLP (512→256→128→2) | Random init | **Yes — PPO** |

- `_pre_decoder_embedding` is accessed via `self.actor.attention._pre_decoder_embedding` after each forward pass.
- The backbone forward pass is run with `torch.no_grad()` during switcher training.
- Only the **Switcher Head** parameters are passed to the PPO optimizer.

---

## 5. Workspace Architecture

```
robot_nav/models/MARL/capswitcher/
│
├── capswitcher_project_summary.md     # This document
│
├── config/
│   ├── default.yaml                   # Default hyperparameters
│   ├── env_6robot.yaml                # 6-robot environment config
│   └── train_capswitcher.yaml         # CAPSwitcher PPO training config
│
├── policies/
│   ├── coarse_steering.py             # Least-squares and nonlinear coarse steering
│   │                                  #   - pinv(A_reduced) heading optimization
│   │                                  #   - Nonlinear progress optimization
│   │                                  #   - Free-space validation utilities
│   ├── gat_backbone.py                # GAT backbone wrapper (FROZEN)
│   │                                  #   - Loads AttentionObstacleOptimized weights
│   │                                  #   - Freezes all parameters
│   │                                  #   - Exposes extract_pre_decoder_embedding()
│   │                                  #     → returns _pre_decoder_embedding (B·N, 512)
│   │                                  #   - Exposes get_precise_actions()
│   │                                  #     → runs frozen actor_target for precise mode
│   └── cap_switcher.py                # CAPSwitcher Switcher Head (TRAINED)
│                                      #   - Input: _pre_decoder_embedding (B·N, 512)
│                                      #   - Reshape → (B, N, 512)
│                                      #   - Mean pool over robots → (B, 512)
│                                      #   - MLP(512→256→128→2) → logits
│                                      #   - Categorical → {0=coarse, 1=precise}
│
├── rl/
│   ├── switcher_env.py                # SwitcherEnv: Gym-like wrapper
│   │                                  #   - reset(): run sim, call GAT backbone,
│   │                                  #     return _pre_decoder_embedding
│   │                                  #   - step(action): execute coarse or precise
│   │                                  #     for selection_interval steps
│   │                                  #   - Reward: 
│   ├── switcher_ppo.py                # PPO trainer
│   │                                  #   - SwitcherActorCritic (actor + value head)
│   │                                  #   - SwitcherRolloutBuffer
│   │                                  #   - train(): clipped surrogate + entropy bonus
│   └── reward.py                      # Reward shaping constants and functions
│
├── experiments/
│   ├── phase1_freespace.py            # Validate coarse steering in free space
│   │                                  #   - Compare least-squares vs. nonlinear
│   │                                  #   - Measure forward progress per step
│   ├── phase2_learned.py              # Trained CAPSwitcher full evaluation
│
├── utils/
│   ├── geometry.py                    # Angle wrapping, circular mean, heading utils
│   ├── linalg.py                      # Pseudoinverse, rank analysis, null space
│   └── logging.py                     # Experiment logging and checkpointing
│
└── checkpoints/
    ├── gat_pretrained/                # Pre-trained TD3Obstacle weights (actor + critic)
    │                                  #   Used by gat_backbone.py (frozen)
    └── cap_switcher/                  # Trained CAPSwitcher Switcher Head checkpoints
```

### Key File Responsibilities

| File | Frozen? | What it does |
|---|---|---|
| `gat_backbone.py` | Yes (all weights) | Wraps `AttentionObstacleOptimized`; exposes `_pre_decoder_embedding` and precise actions |
| `cap_switcher.py` | No (trained) | Switcher Head: mean-pool + MLP → {coarse, precise} |
| `coarse_steering.py` | N/A (no weights) | Least-squares `t* = pinv(A)·dθ` and nonlinear optimizer |
| `rl/switcher_env.py` | N/A | Environment loop; calls backbone + either coarse or precise executor |
| `rl/switcher_ppo.py` | No (trained) | PPO update loop for Switcher Head only |
