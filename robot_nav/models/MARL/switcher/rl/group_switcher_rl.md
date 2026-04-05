# Group Switcher — RL Architecture Reference

## Overview

The group switcher uses a PPO-trained policy to select which candidate group of robots to activate at each decision step. The system consists of three main components: a feature builder, an actor-critic network pair, and the PPO training loop.

---

## RLFeatureBuilder — Per-Group Feature Vector

For each candidate group $G_k$ ($k = 1, \ldots, M$), construct a feature vector $\mathbf{x}_k \in \mathbb{R}^D$ where $D = 2 \times 512 + 12 = 1036$:

$$\mathbf{x}_k = \bigl[\; h_{G_k}\;,\; h_{\text{glob}}\;,\; \mathbf{s}_k \;\bigr]$$

- **$h_{G_k} \in \mathbb{R}^{512}$** — Group embedding: mean-pool of member embeddings $\{h_i : i \in G_k\}$
- **$h_{\text{glob}} \in \mathbb{R}^{512}$** — Global embedding: mean-pool of all 14 robot embeddings (same for every group, broadcast)
- **$\mathbf{s}_k \in \mathbb{R}^{12}$** — 12 scalar features (see table below)

### Per-Group Scalar Features ($\mathbf{s}_k$)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `size_feat` | $\|G_k\|/7$ (normalized group size) |
| 2 | $A_{\text{in}}$ | Mean intra-group attention from $A_{rr}$ |
| 3 | $A_{\text{out}}$ | Mean group-to-outside attention from $A_{rr}$ |
| 4 | $A_{\text{obs}}$ | Mean group-to-obstacle attention from $A_{ro}$ |
| 5 | `mean_dist_goal` | Mean distance-to-goal of group members |
| 6 | `min_dist_goal` | Min distance-to-goal in group |
| 7 | `min_clearance` | Worst obstacle clearance in group |
| 8 | `frac_reached` | Fraction of group members already at goal |
| 9 | `mean_heading_err` | Mean $\|\theta_i - \text{atan2}(\Delta y, \Delta x)\|$ |
| 10 | `var_dist_goal` | Variance of distance-to-goal across all 14 robots (global, broadcast) |
| 11 | `frac_reached_global` | Global fraction of robots reached (broadcast) |
| 12 | `steps_elapsed_frac` | Current step / max episode steps (broadcast) |

---

## State Feature Vector (Critic only)

$$\mathbf{s}_{\text{state}} = \bigl[\; h_{\text{glob}}\;,\; \mathbf{c} \;\bigr] \in \mathbb{R}^{517}$$

- $h_{\text{glob}}$ — 512-dim global embedding (same as above)
- $\mathbf{c}$ — 5-dim state-level scalars: `mean_dist_goal_all`, `var_dist_goal_all`, `frac_reached_global`, `min_clearance_all`, `steps_elapsed_frac`

---

## PPO Group Switcher Networks

### Actor Network (Two-Tower Fusion → Categorical Policy)

**Input:** group feature matrix $X \in \mathbb{R}^{M \times 1036}$ (one row per candidate group).

For each group row $\mathbf{x}_k$:

1. **Embedding Tower:**
   - Input: $[h_{G_k} \| h_{\text{glob}}] \in \mathbb{R}^{1024}$
   - Linear(1024 → 256) → GELU → LayerNorm(256)
   - Output: $\mathbf{e}'_k \in \mathbb{R}^{256}$

2. **Scalar Tower:**
   - Input: 12-dim scalar vector
   - Linear(12 → 64) → GELU → LayerNorm(64)
   - Output: $\mathbf{s}'_k \in \mathbb{R}^{64}$

3. **Fusion:**
   - Input: $[\mathbf{e}'_k \| \mathbf{s}'_k] \in \mathbb{R}^{320}$
   - Linear(320 → 256) → GELU → LayerNorm(256) → Dropout(0.1) → Linear(256 → 1)
   - Output: logit $z_k \in \mathbb{R}$

4. **Categorical distribution:**
   $$\pi(G_k \mid s) = \text{softmax}([z_1, \ldots, z_M])_k$$
   Sample group index $k^* \sim \text{Categorical}(\pi)$

### Critic Network (Value Head)

**Input:** state feature $\mathbf{s}_{\text{state}} \in \mathbb{R}^{517}$

1. **State Embedding Tower:** Linear(512 → 128) → GELU → LayerNorm(128)
2. **State Scalar Tower:** Linear(5 → 32) → GELU → LayerNorm(32)
3. **Fusion:** $[\cdot \| \cdot] \in \mathbb{R}^{160}$ → Linear(160 → 64) → GELU → LayerNorm(64) → Linear(64 → 1)
4. **Output:** $\hat{V}(s) \in \mathbb{R}$

---

## PPO Training Hyperparameters

- **Objective:** Clipped surrogate with $\epsilon = 0.15$
- **GAE:** $\gamma = 0.99$, $\lambda = 0.95$
- **Entropy regularization:** Annealed $0.08 \to 0.01$ over 60% of training
- **Actor LR:** $5 \times 10^{-5}$
- **Critic LR:** $10^{-3}$
- **Rollout buffer:** 256 switcher decisions per update, 4 PPO epochs