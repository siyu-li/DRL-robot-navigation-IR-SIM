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

When group s is activated with rotation parameter t, each robot i rotates by dθ\_i = A[i, s] · t. Then the group's **members** drive forward along their individual headings by a fixed distance d (non-members hold still). The total rotation from activating multiple groups with parameters **t** = [t1, t2, ..., t_K] is **dθ = A · t**.

### The Rank-Deficiency Problem

The actuation matrix **A** is rank-deficient: it has fewer independent columns than robots. This means the system cannot independently control all robot headings simultaneously. Only a low-dimensional subspace of the full rotation space is reachable through any combination of group activations. For the simplified 6-robot system, we artificially constrain to rank 2 (by using only 2 of 3 groups) to study this limitation.

### Two Control Regimes

**Coarse group control.** A coarse control of a chosen group has two physical phases. (1) **Rotation:** all robots rotate according to the actuation matrix (robots outside the group membership rotate by t, group members do not). (2) **Move:** only the *members* of the chosen group advance forward by a fixed translation length; non-members hold still. This is physically efficient — the moving robots make forward progress with no reversal or back-and-forth. However, due to rank deficiency of A, the rotation cannot independently steer each robot. The swarm can only adjust headings within the low-dimensional controllable subspace. Obstacle avoidance is not possible at this level.

> Implementation note: in code (`coarse_steering.py`) each phase is split into velocity-bounded **sub-step frames** so a large rotation/translation is realised over several simulator steps within the angular/linear-velocity caps. The reduced actuation matrix is rebuilt per call (the chosen group's own column is dropped, then one of the remaining columns is dropped at random) to give an artificially rank-2 system.

**Precise (dense) group control.** Individual robots can be steered independently. Conceptually this composes multiple group activations (e.g. activating a coarse group, then additional groups that rotate unwanted robots by 180° and move them backward to cancel their motion), which is physically wasteful because of the back-and-forth reversal. In implementation, precise control is realised by the frozen Graph Attention Network (GAT) policy run **sequentially, one robot at a time**: each robot is driven by its individual GAT navigation action (with obstacle avoidance) for a few sub-steps while all other robots hold still. Resolving robots one-by-one is the source of the physical/time inefficiency the switcher must weigh against coarse control.

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

The key prerequisite (and a contribution of this work) is a viable coarse group steering mechanism. We replace the failed minimum-action approach with steering over the rank-deficient actuation matrix. Both methods are implemented (`CoarseSteering.method`); given desired robot headings toward the goal, we solve either:

```
least_squares:  t* = pinv(A_reduced) · dθ_desired        (closest reachable headings)
nonlinear:      t* = argmax  Σ_{i∈members} Δdistance_to_goal_i(t)   (location-progress, BFGS)
```

and apply `dθ_actual = A_reduced · t*` to **all** robots. The chosen group's **members** then translate forward by the fixed `move_distance`. `A_reduced` is the artificially rank-2 actuation matrix described in §1 (chosen group's column dropped, plus one random drop). The nonlinear variant is initialised at the least-squares solution; `last_solve_time_s` records the per-call solve time for the LS-vs-nonlinear runtime comparison (Objective 1). This produces meaningful collective motion despite rank deficiency, giving the switcher a coarse control option worth selecting.

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

  self_embed = robot_embed.reshape(B·N, 256)
  _pre_decoder_embedding = cat(self_embed, attn_out)  → (B·N, 512)
  Stored as self._pre_decoder_embedding (detached) — selectable alt tap.

Stage 4 — Decoder / Task Head               (USED as switcher input)
  decode_1 : Linear(512 → 512) + LeakyReLU
  decode_2 : Linear(512 → 512) + LeakyReLU
  → att_embedding (== attn_out == H) : (B·N, 512)
  ── CUT POINT (default) ─────────────────────────────────
  The per-robot decoder output H is the switcher input
  (embedding_source="decoder").  "pre_decoder" remains a
  selectable alternative tap for ablation.
  ── CUT POINT ──────────────────────────────────────────

Policy Head (in ActorObstacle)              (DISCARD for switching)
  3-layer MLP: 512→400→300→action_dim + Tanh
  (still used to produce the precise-mode navigation actions)
```

### 4.2 Why `_pre_decoder_embedding` Is the Right Cut Point

| Component | Encodes | Useful for switching? | Decision |
|---|---|---|---|
| Node encoder (emb1/2) | Generic per-robot local state | Yes — same input space | **Freeze** |
| Hard attention (hard_mlp / hard_mlp_obs) | Which neighbors/obstacles are in proximity | **Directly yes** — captures "am I near an obstacle?" | **Freeze** |
| Soft message passing (q/k/v, attn_score) | Relational importance of neighbors | Yes — spatial context | **Freeze** |
| `_pre_decoder_embedding` (512-dim) | Self state ⊕ aggregated neighborhood | Yes — selectable alternative tap | **Freeze; available via `embedding_source="pre_decoder"`** |
| Decoder output `attn_out`/`H` (512-dim) | Per-robot navigation-context features | **Yes — default switcher input** | **Freeze & use as input (`embedding_source="decoder"`)** |
| Policy head | Maps to lin/ang velocity | Navigation-specific | **Discard for switching (still used for precise actions)** |

> **Note (revised):** the default switcher input is the **per-robot decoder output** `H` (`attn_out`), not the pre-decoder embedding. Because the GAT and TD3 policy were trained *jointly*, neither tap is truly task-agnostic; the decoder output additionally encodes "what the navigation policy wants to do here", which is itself diagnostic of when precise control is needed. `embedding_source` makes the tap configurable for ablation. Crucially, the per-robot embeddings are **kept unpooled** — the switcher head learns its own permutation-invariant aggregation (Deep Sets), because the per-robot navigation embeddings were never trained to be summarizable under a fixed pool.

### 4.3 Updated CAPSwitcher Architecture

```
                        GAT Backbone (ALL FROZEN)
                        ┌─────────────────────────────────────────────────┐
robot_obs (B,N,11) ────▶│  Node Encoder → Hard Attention → Soft Message  │
obstacle_obs (B,M,4)    │  Passing → decoder (decode_1/2)                 │
                        │                      ↓                          │
                        │   decoder output H (attn_out)   (B·N, 512)      │
                        └───────────────────┬─────────────────────────────┘
                                            │ detach + reshape (NO pooling)
                                            ▼
                                     (B, N, 512)         Deep Sets Q-Net (TRAINED)
                                            │  ┌────────────────────────────────┐
                                            └─▶│  φ per-robot: 512→256→128       │
                                               │  aggregate: sum ⊕ max  (256)    │
                                               │  ρ: 256→128→2                   │
                                               │  → Q(coarse), Q(precise)        │
                                               │  argmax / ε-greedy  (Double-DQN)│
                                               └────────────────────────────────┘
                                                          │
                        ┌──────────────────────┐          │  ┌────────────────────────────┐
                        │  Coarse Steering     │◀ action=0┘  │  Precise (frozen GAT)      │
                        │  rotate (A·t*) + move│   action=1 ▶│  sequential per-robot actor│
                        │  members; LS / NL    │             │  (others hold still)       │
                        └──────────────────────┘             └────────────────────────────┘
```

### 4.4 Training Setup (Revised)

| Module | Weights | Updated during switcher training? |
|---|---|---|
| `AttentionObstacle` (all 4 stages) | Pre-trained TD3Obstacle actor | **No — fully frozen** |
| `ActorObstacle.policy_head` | Pre-trained TD3Obstacle | **No — used only for precise actions** |
| `DeepSetsQNet` (φ 512→256→128, sum⊕max, ρ→2) | Random init | **Yes — Double-DQN** |

- The per-robot decoder output `H` is obtained from a single frozen forward (`embedding_utils.extract_embeddings_and_actions`, which returns precise actions **and** `H` together).
- The backbone forward pass is run with `torch.no_grad()` during switcher training; the env caches it so each distinct state is forwarded at most once.
- Only the **`DeepSetsQNet`** parameters are passed to the Adam optimizer; training is **off-policy** (replay buffer of per-robot `(N, 512)` observations), chosen over PPO for sample efficiency on this binary decision.

### 4.5 Reward (decision-level)

The reward is computed **once per switcher decision** (one `SwitcherEnv.step`), *not* summed over the sub-steps a mode expands into. This makes it agnostic to how many sim sub-steps a mode consumes (coarse ≈ 10–14, sequential precise = N×5 = 30 for 6 robots), so sub-step count does not bias the value function. From `rl/reward.py` (`SwitcherReward`):

```
r =  k_p · Σ_i (d_start_i − d_end_i)     # progress, summed over robots (telescoping)
   − Σ_i (cl_penalty_i + obs_penalty_i)  # robot–robot + obstacle proximity, at decision end
   + step_penalty(action)                # coarse = −0.5, precise = −3.0
   + R_collision (−100)  if any collision     # terminal, exclusive of shaping
   + R_allgoal  (+200)   if all reached        # terminal, exclusive of shaping
```

Defaults: `k_p=1.0`, `coarse_penalty=−0.5`, `precise_penalty=−3.0`, `r_collision=−100`, `r_allgoal=+200`. Progress telescopes to `initial − final` total distance over a fixed-goal episode, so mode choice is driven by the step penalties and the collision/clearance terms — exactly the physical-efficiency trade-off the switcher should learn. The per-decision proximity penalties come from `MARL_SIM_OBSTACLE.proximity_penalties()`. (The simulator's `reward_phase=6` is used only for backbone state preparation, not for the switcher reward.)

---

## 5. Workspace Architecture

```
robot_nav/models/MARL/capswitcher/
│
├── capswitcher_project_summary.md     # This document
├── __init__.py                        # Package exports (DQN API)
├── embedding_utils.py                 # Single frozen forward → (precise actions, H, attn)
│                                      #   - extract_embeddings_and_actions(..., embedding_source)
│
├── policies/
│   ├── coarse_steering.py             # Two-phase coarse group steering
│   │                                  #   - rotate all robots: dθ = A_reduced · t*
│   │                                  #     (least_squares pinv OR nonlinear BFGS)
│   │                                  #   - move chosen group's MEMBERS by move_distance
│   │                                  #   - rotation/translation split into sub-step frames
│   │                                  #   - group-dependent random rank-2 reduction
│   ├── gat_backbone.py                # GAT backbone wrapper (FROZEN)
│   │                                  #   - Loads TD3Obstacle actor, freezes all params
│   │                                  #   - get_embedding_and_actions() → (raw_actions, H)
│   │                                  #   - embedding_source: "decoder" (default) | "pre_decoder"
│   ├── deep_sets_head.py              # DeepSetsHead: φ → (sum⊕max) → ρ  (reusable readout)
│   └── cap_switcher.py                # SwitcherHead — LEGACY mean-pool MLP (unused by DQN)
│
├── rl/
│   ├── switcher_env.py                # SwitcherEnv: Gym-like wrapper (DQN)
│   │                                  #   - obs: per-robot (N, 512) decoder output (unpooled)
│   │                                  #   - step(action): coarse (group frames) OR precise
│   │                                  #     (sequential per-robot GAT); budget in DECISIONS
│   │                                  #   - decision-level reward via SwitcherReward
│   │                                  #   - forward-pass cache (1 forward / distinct state)
│   ├── switcher_dqn.py                # Double-DQN trainer
│   │                                  #   - DeepSetsQNet (per-robot → sum⊕max → Q-values)
│   │                                  #   - ReplayBuffer of per-robot (N, 512) obs
│   │                                  #   - ε-greedy, target net, smooth-L1 TD loss
│   └── reward.py                      # SwitcherReward — decision-level reward (see §4.5)
│
└── runs/                              # TensorBoard logs
```

> Not present (planned in earlier drafts but not in the tree): `config/`, `experiments/`, `utils/`, `checkpoints/`, `rl/switcher_ppo.py`. The training entry point is `robot_nav/marl_train_capswitcher.py`; checkpoints are written under `checkpoints/cap_switcher/` at save time.

### Key File Responsibilities

| File | Frozen? | What it does |
|---|---|---|
| `gat_backbone.py` | Yes (all weights) | Wraps the TD3Obstacle actor; `get_embedding_and_actions` returns per-robot decoder output `H` + precise actions |
| `deep_sets_head.py` | No (trained) | Permutation-invariant readout: per-robot φ → sum⊕max → ρ (reused by `DeepSetsQNet`) |
| `cap_switcher.py` | No | **Legacy** mean-pool MLP head; superseded by the Deep Sets DQN, kept for reference |
| `coarse_steering.py` | N/A (no weights) | Two-phase rotate (`A·t*`, LS/nonlinear) + move-members coarse control, split into sub-step frames |
| `rl/switcher_env.py` | N/A | Environment loop; coarse (group frames) or sequential precise; decision-level reward |
| `rl/switcher_dqn.py` | No (trained) | Double-DQN update loop for `DeepSetsQNet` only |
