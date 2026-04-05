# Group Switcher — Supervised Ranking Reference

## Overview

The supervised group switcher learns to rank candidate robot groups using oracle-derived labels. It consists of three components: a **`GroupFeatureBuilder`** that constructs per-group feature vectors from GAT embeddings, attention weights, and task-level scalars; a **`GroupSwitcher`** two-tower fusion network that scores each candidate group; and a suite of **ranking losses** (pairwise logistic, hinge, listwise softmax, scheduled margin) for training.

---

## GroupFeatureBuilder — Per-Group Feature Vector

For each candidate group $G_k$ ($k = 1, \ldots, M$), construct a feature vector $\mathbf{x}_k \in \mathbb{R}^D$ where $D = d + d_g + S$:

$$\mathbf{x}_k = \bigl[\; h_{G_k}\;,\; h_{\text{glob}}\;,\; \mathbf{s}_k \;\bigr]$$

- **$h_{G_k} \in \mathbb{R}^{d}$** — Group embedding: pooled (mean or max) over member embeddings $\{h_i : i \in G_k\}$. Default $d = 512$.
- **$h_{\text{glob}} \in \mathbb{R}^{d_g}$** — Global embedding: mean-pool of all robot embeddings (same for every group, broadcast). Defaults to $d_g = d$.
- **$\mathbf{s}_k \in \mathbb{R}^{S}$** — Scalar features.

With the default configuration (`switcher_config.yaml`), $d = d_g = 512$ and $S = 12$, giving $D = 1036$.

### Scalar Features ($\mathbf{s}_k$)

$$S = S_{\text{base}} + |\texttt{extra\_group}| + |\texttt{extra\_global}|$$

With defaults: $S = 4 + 5 + 3 = 12$.

#### Base Scalars (always-on when `base_scalars: true`, 4 dims)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `size_feat` | $\|G_k\| / \texttt{max\_group\_size}$ (default max = 7) |
| 2 | $A_{\text{in}}$ | Mean intra-group attention (excluding self) from $A_{rr}$ |
| 3 | $A_{\text{out}}$ | Mean group-to-outside-robots attention from $A_{rr}$ |
| 4 | $A_{\text{obs}}$ | Mean group-to-obstacle attention from $A_{ro}$ |

#### Per-Group Extra Scalars (`extra_group`, 5 dims with defaults)

| # | Feature | Key | Aggregation | Description |
|---|---------|-----|-------------|-------------|
| 5 | `mean_dist_goal` | `dist_to_goal` | mean | Mean distance-to-goal of group members |
| 6 | `min_dist_goal` | `dist_to_goal` | min | Min distance-to-goal in group |
| 7 | `min_clearance` | `clearance` | min | Worst obstacle clearance in group |
| 8 | `frac_reached` | `reached` | mean | Fraction of group members already at goal |
| 9 | `mean_heading_err` | `heading_error` | mean | Mean heading error in group |

#### Global Context Scalars (`extra_global`, 3 dims with defaults)

| # | Feature | Key | Description |
|---|---------|-----|-------------|
| 10 | `var_dist_goal` | `var_dist_to_goal` | Distance variance across all robots (broadcast) |
| 11 | `frac_reached_global` | `frac_reached_global` | Global fraction of robots reached (broadcast) |
| 12 | `steps_elapsed_frac` | `steps_elapsed_frac` | Current step / max steps, time pressure (broadcast) |

---

## GroupSwitcher — Two-Tower Fusion Network

### Architecture

**Input:** group feature matrix $X \in \mathbb{R}^{M \times D}$ (one row per candidate group).

For each group row $\mathbf{x}_k$:

1. **Embedding Tower:**
   - Input: $[h_{G_k} \| h_{\text{glob}}] \in \mathbb{R}^{2d}$ (default $2 \times 512 = 1024$)
   - `Linear(1024 → 256) → GELU → LayerNorm(256)`
   - Output: $\mathbf{e}'_k \in \mathbb{R}^{256}$

2. **Scalar Tower:**
   - Input: $\mathbf{s}_k \in \mathbb{R}^{S}$ (default $S = 12$)
   - `Linear(12 → 32) → GELU → LayerNorm(32)`
   - Output: $\mathbf{s}'_k \in \mathbb{R}^{32}$

3. **Fusion:**
   - Input: $[\mathbf{e}'_k \| \mathbf{s}'_k] \in \mathbb{R}^{288}$
   - `Linear(288 → 256) → GELU → LayerNorm(256) → Dropout(0.1) → Linear(256 → 1)`
   - Output: logit $z_k \in \mathbb{R}$

4. **Group selection:**
   - **Deterministic:** $k^* = \arg\max_k z_k$
   - **Stochastic:** $k^* \sim \text{Categorical}\bigl(\text{softmax}([z_1, \ldots, z_M] / \tau)\bigr)$

### Weight Initialization

All `Linear` layers use Kaiming uniform initialization (ReLU nonlinearity gain) with zero bias.

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 512 | Dimension of per-robot GAT embeddings |
| `scalar_dim` | 4 | Number of scalar features |
| `embed_hidden` | 256 | Hidden dim for embedding tower |
| `scalar_hidden` | 32 | Hidden dim for scalar tower |
| `fusion_hidden` | 256 | Hidden dim for fusion layer |
| `dropout` | 0.1 | Dropout probability in fusion |

---

## Ranking Losses

### `pairwise_logistic_ranking_loss` (RankNet-style)

For each pair $(i, j)$ where group $i$ should rank above group $j$:

$$\mathcal{L}_{\text{pair}} = \frac{1}{K} \sum_{(i,j)} \log\bigl(1 + e^{-(z_i - z_j)}\bigr)$$

### `hinge_ranking_loss`

$$\mathcal{L}_{\text{hinge}} = \frac{1}{K} \sum_{(i,j)} \max\bigl(0,\; m - (z_i - z_j)\bigr)$$

where $m$ is the margin (default 1.0).

### `listwise_softmax_loss`

Treats group selection as multi-class classification:

$$\mathcal{L}_{\text{list}} = -\log \frac{e^{z_{k^*}}}{\sum_k e^{z_k}}$$

where $k^*$ is the target (oracle-best) group index.

### `RankingLossWithScheduledMargin`

Wraps `hinge_ranking_loss` with a linearly annealed margin:

$$m(t) = m_0 + \frac{t}{T_{\text{warmup}}} (m_f - m_0), \quad t \le T_{\text{warmup}}$$

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_margin` | 0.1 | Starting margin $m_0$ |
| `final_margin` | 1.0 | Target margin $m_f$ |
| `warmup_steps` | 1000 | Steps $T_{\text{warmup}}$ to reach final margin |

---

## Pair Construction Utilities

### `build_pairs_from_scores(scores, margin=0.0)`

Generates all pairs $(i, j)$ where $\text{scores}[i] > \text{scores}[j] + \text{margin}$.

### `build_pairs_from_ranking(ranking)`

Generates pairs from an ordered list of group indices (best-first). Creates $(r_i, r_j)$ for all $i < j$.

---

## Evaluation Metrics

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_ranking_accuracy(logits, pairs)` | `float` | Fraction of pairs where $z_{\text{pos}} > z_{\text{neg}}$ |
| `compute_top1_accuracy(logits, target_idx)` | `bool` | Whether $\arg\max(z) = \text{target}$ |

---

## Configuration

Feature composition is controlled by `switcher_config.yaml` (shared with the RL variant). The relevant keys for the supervised module are:

| YAML Key | Type | Effect |
|----------|------|--------|
| `base_scalars` | `bool` | Include/exclude the 4 base scalars (size, attention stats) |
| `extra_group` | `list[[key, agg]]` | Per-group extra features; `agg` ∈ {mean, min, max, sum} |
| `extra_global` | `list[key]` | Global broadcast scalars |

`GroupFeatureBuilder.from_config(cfg)` accepts a `SwitcherScalarConfig` (loaded via `config_loader.load_switcher_config()`) and wires all feature lists automatically. Changing this config alters the scalar dimension and requires retraining.

---

## Module Files

| File | Contents |
|------|----------|
| `__init__.py` | Public API — re-exports `GroupFeatureBuilder`, `GroupSwitcher`, all loss functions and metrics |
| `feature_builder.py` | `GroupFeatureBuilder` class, `_BASE_SCALAR_DIM` constant |
| `switcher_net.py` | `GroupSwitcher` two-tower fusion network |
| `rank_losses.py` | Loss functions, pair builders, accuracy metrics, `RankingLossWithScheduledMargin` |

---

## Usage

```python
from robot_nav.models.MARL.switcher.supervised import (
    GroupFeatureBuilder, GroupSwitcher,
    pairwise_logistic_ranking_loss, build_pairs_from_scores,
)

# Build features
fb = GroupFeatureBuilder(embed_dim=512, extra_group=[("dist_to_goal", "mean")],
                         extra_global=["steps_elapsed_frac"])
X = fb(h, groups, h_glob=h_glob, attn_rr=attn_rr, attn_ro=attn_ro, extra=extra)

# Score groups
net = GroupSwitcher(embed_dim=512, scalar_dim=fb.scalar_dim)
logits = net(X)

# Training
pairs = build_pairs_from_scores(oracle_scores, margin=0.0)
loss = pairwise_logistic_ranking_loss(logits, pairs)

# Inference
best_group = net.select_group(logits, mode="argmax")
```

Training entry point: `python -m robot_nav.scripts.train_switcher`
