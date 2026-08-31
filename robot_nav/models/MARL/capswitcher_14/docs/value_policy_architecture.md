# Value and policy network architecture

Companion to `value_data_collection_plan.md`. That doc covers what data to collect;
this one covers what to train on it. Scope: original system, legacy uncoupled
rotation or precise-coupled pinv, precise-all action, config A only.

---

## 0. Premises this design rests on

Three measured findings drive every choice below. If any of them turns out wrong,
revisit the design rather than patching it.

**0.1 Guidance-limited, not budget-limited.** GAZ success is flat at 76/77/77%
across budgets 100/200/400 while search effort triples. PHS reaches 89% and A* 85%
with the same model, action set, and learned `h`. The ceiling is the guidance, and
~90% is reachable with this action set.

**0.2 The analytic `alpha` is badly inadmissible, with a state-dependent sign.**
`alpha = precise_unit / (lin_max * step_time) ~= 77` per robot-meter is the *serial*
rate — the price of moving one robot at a time. The coarse table delivers robot-meters
at 1.26 (size-7 groups) to ~18 (size-3/4), because a size-7 action buys seven robots'
meters for one action's cost. So `alpha * sum(d)` overprices coarse-transport states
by up to ~60x while being roughly exact-to-under in precise endgames.

There is no single scalar fix: the error is a **state-dependent exchange rate**, not
an offset. This is the entire argument for a multiplicative value form.

Counter-term to keep in view: `sum(d)` is itself a *lower* bound on required
robot-meters, since obstacles force detours. The two errors run in opposite
directions; the measured net overestimate means the `alpha` error dominates in most
states. Both must be absorbed by the learned rate.

**0.3 The two consumers stress different properties.** A* compares `f = g + h`
across the whole open list, so it consumes calibration and punishes state-correlated
bias. Gumbel AZ compares Q among children of a common parent and normalizes, so
sibling-shared bias cancels and it consumes discrimination. Measured: analytic beats
learned under A*, learned beats analytic under GAZ. Calibration strictly contains
ranking, so train for calibration and regularize for ranking — that value serves both
searches; a rank-only value serves one.

**0.4 Rendered failure modes.** (a) Arrived-robot jiggle — actions that drag
already-arrived robots off their goals. (b) Robot-robot blocking, especially an
arrived robot parked on another robot's goal ray, which is a permanent obstacle under
this action set. These two dictate the feature set and the head decomposition.

---

## 1. Shared trunk

One trunk, two heads. Both networks need the same thing from the state: per-robot
"how much work is left for me, and who is in my way."

**Weight sharing:** do NOT share trunk weights between value and policy. Train the
value first from A*/PHS traces; a shared trunk means round-0 policy CE gradients
perturb the value you just gated on. Same architecture, separate weights.

**No transfer from the navigation GAT.** Considered and rejected. The transferable
part would have been the geometric relevance kernel (weight decays with distance,
rises with angle match), but nav relevance is short-horizon — "will this collide with
me next step" — while value relevance is long-horizon: an arrived robot parked 3 m
away on my goal ray is nearly irrelevant to one step and dominant in cost-to-go. Same
geometry, opposite relevance. The transferred kernel would anchor the length scale in
the wrong regime. Random init throughout.

### 1.1 Node features (~24 dims)

All lengths normalized by arena diagonal `L`; all clearances capped.

| Group | Features |
|---|---|
| Progress | `d_i/L`, `arrived_i = 1[d_i < eps]` |
| Direction | `cos phi_i`, `sin phi_i` (bearing to own goal) |
| Kinematics | `cos dtheta_i`, `sin dtheta_i` where `dtheta_i = theta_i - phi_i` |
| Position | `p_i/L`, `g_i/L` relative to arena center |
| Obstacle | free-path along goal ray `/d_i`, min ray clearance, static clearance at `p_i`, at `g_i` |
| Shareability | `n_aligned_i/N`, mean and max `cos(phi_i - phi_j)` over non-arrived `j` |

`free_path_i / d_i` is the single most informative detour feature — "is my straight
line actually available." The shareability block is what the `alpha`-overprice term
needs: it says how many other robots can be transported in the same direction.

No positional encoding, no robot index. Permutation equivariance is what allows
transfer from N=14 to N=30.

### 1.2 Typed edges

Three relation types, three separate edge encoders. All geometry expressed in robot
`i`'s goal-ray frame, where `s_ij` = along-ray component and `l_ij` = lateral.

| Type | Neighbors | Features |
|---|---|---|
| `robot` | non-arrived robots | `norm(dp)/L`, `(s_ij, l_ij)`, `on_ray_ij`, `cos(phi_i - phi_j)`, `cos(dtheta_i - dtheta_j)`, `co_group_ij/22` |
| `arrived` | arrived robots | same, plus `1[norm(p_j - g_i) < eps]` (parked on my goal) |
| `obstacle` | k-NN, k ~= 8 | `norm(dp)/L`, `(s, l)`, bearing `(cos, sin)`, extent/radius, `blocks_ray` |

`on_ray_ij = 1[abs(l_ij) < w_corr and 0 < s_ij < d_i]`.

Notes:
- `arrived` gets its own type rather than a flag on `robot`, because "cannot be moved
  aside" is failure mode 0.4(b) and deserves its own learned kernel.
- `co_group_ij` (how many of the 22 coarse groups contain both robots) is how the
  fixed group structure reaches the *value* net. It is what lets the rate head learn
  "these two can be transported together, so their meters are cheap."
- Obstacles are typed edges, not per-robot scalars, because detour cost depends on
  *where* the obstacle is, not just how close. Obstacles have no node state — the
  message from an obstacle edge is edge-content only.
- Pairwise relations must be explicit inputs, not left for attention to discover.
  Dot-product attention compares content, and two robots can have near-identical
  tokens while one blocks the other. The relation lives in the pair.

### 1.3 Trunk layer

3 rounds. Per round, per head, for edge type `tau`:

```
a_ij  = softmax_j [ MLP_bias^tau(e_ij) + (W_q h_i) . (W_k^tau h_j) / sqrt(d) ]
m_i   = sum_tau sum_{j in N_tau(i)} a_ij * (W_v^tau h_j + W_m^tau e~_ij)
h_i  <- h_i + W_o m_i
h_i  <- h_i + FFN(h_i)                                    # pre-norm
```

The `MLP_bias` term is load-bearing: it lets "j is arrived and sitting in my
corridor" force attention onto j regardless of embedding similarity. Without it this
degenerates into a set encoder that averages neighbors and never learns blocking.

Plus one global CLS token attending over all robots, carrying `N`, `n_arrived/N`,
`sum(d)/L`, mean clearance, obstacle density, available group-size counts.

**Sizes:** `d_model = 64`, 4 heads, `d_e = 32`, FFN 2x. ~35k params per round,
~125k total for the value net. Try `d_model = 128` only if the 64 version underfits
on train.

**Three rounds, not one.** Blocking is a chain: A is blocked by B, B cannot move
aside because arrived C is beside it. A single pool-and-concat readout gives A raw
geometry about B but never lets A learn that B is itself stuck. 2-3 hops minimum.

**v0 fallback worth benchmarking:** a plain edge-MLP GNN
(`m_i = sum_j MLP([h_i, h_j, e_ij])`, sum-aggregate) has no attention parameters and
may be more sample-efficient at this label scale. Run it first; attention must earn
a held-out margin.

---

## 2. Value head (state-only)

```
V(s) = sum_i [ d_i * alpha * exp(r_i) + b_i ]

r_i = w_r . h_i          clamped to [log(c_min/alpha), 2] = [-4.11, 2]
b_i = softplus(w_b . h_i)
```

Multiplicative in log-rate space, because per 0.2 the revealed structure is a
state-dependent cost-per-robot-meter over ~[1.26, 77+]. A 60x range is a +-4-nat
swing in log space — well-conditioned. An additive residual would have to output
`-0.98 * alpha * d_i` in transport states and near-zero-or-positive in endgames:
same network, opposite-signed outputs of wildly different magnitude, with a
state-determined crossover. That is bad conditioning, not inductive bias.

The rate head is legible: `r_i` answers "how efficiently can this robot's remaining
meters be transported?" Cheap when many robots share its direction; expensive when
blocked, isolated, or in a precise endgame.

**Initialization and clamping — these matter more than they look:**

- Zero-init the rate head (`w_r = 0`, bias 0) so `exp(r_i) = 1` exactly and the
  untrained net *is* `alpha * sum(d)`. Init and fallback are the same object.
- Init the additive head bias to `-6` so `softplus(-6) ~= 0.0025`, else you start
  with +0.69 spurious cost per robot.
- Hard clamp the rate, not a soft floor — a softplus-style floor perturbs the `r=0`
  point and destroys exact recovery.
- Upper cap `r_max = 2` because `sum(d)` undercounts meters (detours), so a robot can
  legitimately exceed the precise rate — but not by 7x.
- The additive head MUST be non-negative, or it eats through the floor.

**What the clamp does and does not buy.** `exp(r_i) >= c_min/alpha` guarantees
`h >= c_min * sum(d)` — a *floor*, not a ceiling. This is NOT admissibility of the
learned `h`; admissibility needs an upper bound and nothing here provides one. What
it buys is **non-collapse**: the net can never drive the value to zero and turn the
search into blind BFS. Do not claim admissibility for the learned `h` in the paper.

`c_min * sum(d)` on its own *is* a valid admissible heuristic (each robot travels at
least its straight-line distance; no robot-meter costs less than the cheapest coarse
rate). If the paper needs an admissible baseline for the A*/PHS comparison, that is
it — not `alpha * sum(d)`.

**Division of labor:** rates carry distance-proportional cost (blocking inflates
them); the additive head carries distance-independent cost (waiting, yielding,
jiggle near arrival) that a purely multiplicative-in-`d` form cannot express for a
nearly-arrived-yet-blocked robot. This mirrors the two failure modes in 0.4.

---

## 3. Policy head (action-conditioned)

Group membership is a constant matrix `M in {0,1}^{22x14}`.

```
u_ik = prescribed unit direction for robot i under group k
z_k  = AttnPool_{i in G_k} MLP([ h_i || u_ik || cos(u_ik, phi_i) || clearance along u_ik ]) || s_k
l_k  = MLP_pi(z_k)
```

One **shared** `MLP_pi` across all 22 groups, not 22 heads — that is what generalizes
across group identity and survives a different group table at N=30.

Action-conditioning belongs here and NOT on the value. Rationale: the forward model
for coarse actions is exact and cheap (one pseudoinverse), so
`Q(s,a) = c(s,a) + V(s')` is directly computable. An action-conditioned value would
learn to approximate a computation already available exactly — the same trap the
abandoned deep-Q attempt fell into. On the policy, by contrast, `u_ik` is strictly
more informative than pooling `h_i` alone: it says per robot whether the prescribed
motion is aligned, orthogonal, or backwards, including for the non-target robots the
rank-deficient solve drags along. That drift is what separates good groups from bad.

**Group scalars `s_k`:**
- circular variance of `{phi_i}_{i in G_k}` — coherence of member goal directions.
  Single most important scalar: it is the "can these be transported together" quantity
  that 0.2 identified as missing physics.
- `n_arrived(G_k)/abs(G_k)` — direct handle on the jiggle failure mode
- mean/min/max member `d_i`, mean `abs(dtheta_i)`, min member clearance, min ahead-clearance

**Precise action:** separate head off the CLS token plus mean/max pools. It is a
different action *type* (all robots, serial execution, obstacle-capable); a shared
head over a size-14 "group" models the wrong thing.

**Verification-cost caveat:** computing `u_ik` needs the pseudoinverse per group, so
this prior cannot prune before paying verification. Fine at N=14 where all 22 are
verified anyway. Keep a cheaper pre-verification variant (the `s_k` scalars alone)
behind a flag for when the group table grows.

**Round-0 targets:** one-hot CE on the chosen edge. NOT advantage-weighted —
advantage weights would come from sibling `g_child + h_child` with `h` the inflated
analytic, so the weighting inherits exactly the inadmissibility measured in 0.2.
Advantage weighting returns in round 1 with the learned value.

---

## 4. Losses

```
L = Huber(V, y_V)
  + lambda_aux  * sum_i Huber(v_i, y_i)
  + lambda_rank * margin( q(s, a+) < q(s, a-) )
```

- `y_V = g(goal) - g(node)` on solved-path nodes — exact in-model remaining cost,
  already in cost units.
- `y_i` = per-robot cost-share attribution (arrival index + cost share), offline
  derivable from solved plans. This is the answer to label scarcity: it multiplies
  effective supervision by ~N and pins the `v_i` decomposition. It is also the reason
  the head is per-robot at all.
- **Rank on `q = c(s,a) + V(s')`, not on `V(s')`.** Sibling actions have different
  step costs; the quantity the search orders is `q`. A pairwise loss on `V` alone
  trains the wrong ordering wherever step costs differ. Verify the collected sub-A*
  labels were built as `c + h*` too.

**Starting weights:** `lambda_aux = 0.3`, `lambda_rank = 1.0`. The ranking term is
tilted up relative to a pure-calibration reading because per 0.2 the absolute labels
carry state-dependent teacher slack, while sibling pairs largely cancel it (both
children's subtrees are planned by the same biased search from nearby states). Treat
the ratio as a real hyperparameter.

**Sub-A\* labels are absolute, not merely ordinal.** When a sub-A* from child `s'`
solves, it returns `h*(s')` exactly — a regression target, landing precisely on the
hard blocked states that on-path collection censors out. Feed solved sub-A* results
into *both* losses. Reserve rank-only treatment for pairs where one side capped out
(there `h* >= cap`, still a valid ordering constraint against a solved sibling).

---

## 5. Data requirements

Count **instances** (distinct map/start/goal seeds), not rows. Twenty nodes along one
solved path are near-duplicates — maybe 3-5 effectively independent samples.

| | Phase 1: is it learnable | Phase 2: deployable |
|---|---|---|
| Distinct instances | 300-500 | 1,500-2,500 |
| Solved plans | ~600-1,000 | ~3,000-5,000 |
| Absolute-label states | 10-20k | 50-100k |
| Sibling pairs | 3-5k | 20-30k |
| Per-robot aux targets | 14x state count | 14x state count |

The naive "125k params needs 1e6 labels" read does not apply: the rate head is
zero-initialized at the analytic baseline, so the net learns a bounded correction in
log space, not cost-to-go from scratch. Effective capacity needed is far below the
parameter count.

**Phase 1 answers exactly one question:** does sibling Spearman beat analytic-`h` on
held-out instances? If not at 500 instances, more data will not fix it — features or
labels are wrong.

**Three collection rules that matter more than the totals:**

1. **Split by instance, never by node.** A random row split leaks catastrophically —
   the same state appears in train and val under trivial perturbation, val loss looks
   excellent, closed-loop does not move. Hold out whole seeds.
2. **Budget sibling pairs.** Each costs a sub-A*. Branch only where analytic `h` ties
   or nearly ties among children, and cap at 3-4 sampled pairs per node rather than
   all `C(k,2)` — pairs from one node are heavily correlated.
3. **Over-sample blocked states.** Uniform collection is mostly easy transport, where
   `alpha * sum(d)` already ranks fine and the net has nothing to learn. Filter toward
   states with non-zero arrived-robot-on-ray count.

**Underfit/data-limited test.** Train at Phase 1 volume, compare train vs val loss.
Val ~= train means underfitting — the fix is capacity or features, not collection.
Train well below val means data-limited — then scale to Phase 2. Run this before
committing to a large collection run; sub-A* labels are the expensive resource.

**Config check:** existing traces in `runs/g1_rebaseline/traces/` are coupled +
pairs/singles on eval seeds 1000-1099 — wrong config and wrong seeds. Training needs
fresh legacy-config collection on seeds 2000+.

---

## 6. Build order

Each step gated before the next.

1. **Trunk + rate head only.** No additive head, no attention (edge-MLP GNN).
   Gate: beat analytic-`h` on sibling Spearman, held out by instance. If this fails,
   the problem is features, not capacity.
2. **Add the additive head.** Check `b_i` actually fires on nearly-arrived-but-blocked
   robots. If it stays ~0 everywhere, the jiggle cost is not in the labels and the
   attribution targets need revisiting.
3. **Swap in edge-biased attention.** Keep only if it earns a held-out margin over
   step 1.
4. **Policy, round-0 one-hot CE.** Separate weights, random init.

**Closed-loop bars — both must be met simultaneously:** >= 89% success (the analytic
baseline) AND ~12k transitions/ep (the learned-geometry baseline). Beating one while
losing the other is the failure mode both prior attempts already demonstrated.

---

## 7. Diagnostics to build now, not later

**Rate-band check.** Dump `exp(r_i)` for every robot in a handful of rendered states.
The expected answer is known a priori: near `1.26/alpha` for robots in coherent
transport, near 1 for endgame robots, high for blocked ones. If the rates do not land
in those bands on states you can eyeball, the net is fitting something other than the
physics derived in 0.2, and no aggregate metric will reveal that.

**Attention kernel heatmap.** Plot `MLP_bias^tau(e)` over distance x relative bearing,
per edge type, after training. Cheap and directly interpretable. If the `arrived` and
`robot` kernels converge to the same shape, the net is not learning the blocking
failure mode. This is also the figure that makes the architecture argument in the
paper without a single equation.

**Sibling Spearman vs analytic**, held out by instance, as the headline offline gate.
Nothing goes closed-loop until it clears.

---

## 8. Open items

- `lambda_rank / lambda_aux` ratio — swept, not assumed.
- `alpha`-scale sweep at collection time: measures how much slack the labels carry and
  may identify a better collection operating point (broader search -> cheaper plans ->
  tighter labels). Load-bearing, not a nicety.
- Whether the analytic baseline's 89% / 35k numbers were measured at `alpha = 77` or
  would change at the admissible scale `c_min`. The sweep answers this.
- Edge-MLP GNN vs edge-biased attention — decided by held-out margin at step 3, not
  assumed.
