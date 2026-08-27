# Value corpus: A*-teacher data collection & training targets

Status: plan, 2026-08-26.  Follows the refocused scope (legacy system:
precise-all, uncoupled — see `value_prior_redesign.md` scope note) and the
design discussion landed in this session:

* train the value **for calibration** (regression on exact trace labels),
  **regularize/measure for ranking** (the property Gumbel-style search
  consumes hardest — learned-beats-analytic under GAZ, analytic-beats-learned
  under A* triangulates this);
* teacher = **A\* with the analytic leaf** (`--value-model` omitted).  Not the
  learned net (its state-correlated bias would launder into the labels), not
  unguided search (branching 23 × depth 30–80 makes h=0 solve nothing at any
  affordable cap);
* value is the primary learning problem (adversarially queried by search);
  prior is a supervised side-product (imitation + advantage weighting, later
  value-distillation).

## 0. The α finding (measured 2026-08-26) and what it changes

`ForwardModel14.cost_to_go` default: `α = precise_unit/(lin_max·step_time)
= 11.57/0.15 ≈ 77.1` cost per robot-meter — the *precise-mode* price of
distance.  The coarse table sells robot-meters at `8.8/7 ≈ 1.26` (size-7) to
`52.7/3–4 ≈ 13–18` (size-3/4).  So `α·Σd`:

* **overestimates** h\* by up to ~60× in states where coarse progress
  suffices (bulk transport phase);
* is ≈ exact-to-underestimating in precise-endgame states (α assumes the GT
  policy drives at `lin_max`; it usually doesn't).

Consequences:

1. **Labels are reference-policy costs, not near-optimal costs.**  A* under a
   strongly inadmissible h behaves like weighted-A*/greedy on Σd; solved-plan
   costs are exact costs of *a good policy's* plan (89% success closed-loop),
   with unknown suboptimality slack.  Regression to them is still
   well-defined and consistent; just don't call the targets optimal.
2. **The closed-set hinge lower bound** (`V(n) ≥ C* − g(n)` for every node
   the solved search touched) is **not rigorous** — it needs C\* optimal.
   Keep it as *soft* supervision with a slack factor λ (§4.3), tuned from the
   pilot's α-sweep evidence.
3. **The residual parameterization loses its one-sided floor.**  `α·Σd` is
   not a lower bound on h\*, so `h = α·Σd + softplus(net)` is wrong.
   Current leaning (2026-08-26): **multiplicative log-rate residual**,
   `v_i = d_i·α·exp(r_i(s))`, `V = Σ v_i + small residual head`.  The true
   structure is a state-dependent cost-per-robot-meter in ~[1.26, 77+]:
   log-rates make the 60× range well-conditioned, `r ≡ 0` still recovers the
   analytic baseline, the per-robot decomposition matches the auxiliary
   labels, and clamping `exp(r_i) ≥ c_min/α ≈ 1/61` (c_min = 1.26, the
   size-7 coarse rate — a *valid* admissible floor) restores a rigorous
   lower-bound guardrail.  The additive term/residual head owns
   distance-independent costs (waiting, yield, jiggle) that the
   multiplicative form cannot express near `d_i → threshold`.
4. The pilot gains an axis: **α-scale sweep** (§2) to measure how much plan
   cost improves when the search broadens — a direct estimate of label
   suboptimality, and possibly a better collection operating point.

(Trivial: the `cost_14robots.yaml` header comment "= 4309.9" is stale — it
matches an older `precise_unit = 61.57`; today 11.57×5×14 = 809.9.)

## 1. Pre-flight (half a day)

1. **Trace round-trip smoke test**: 1 episode with `--log-traces`, then load
   shards, run `on_path_rows` / `value_labels`, verify: path indices resolve,
   `y = plan_cost − g` positive and decreasing along the path, branch counts
   = 23 per expanded node, shard size ≈ expectation (~1.2 KB/node).
2. **`--analytic-alpha-scale` flag** (small change in `eval_gaz14_baselines`):
   wrap the leaf as `analytic_cost_to_go(gd, thr, scale·analytic_alpha())`.
   Record the scale in the trace `meta.json`.
3. **Commit the working tree** before any corpus run (the 2026-08-25 analytic
   rerun was on an uncommitted tree — don't repeat that for training data;
   meta.json carries the code stamp).

## 2. Pilot (10 episodes/cell, ~1 day wall-clock, decision-driving)

Grid, all on seeds 2000–2009, scattered world, legacy config (no
`--coupled-precise`, no `--precise-config`, no `--value-model`,
no `--prior-model`):

| axis | cells |
|---|---|
| cap `--max-transitions` | 5 000 / 20 000 / 50 000 (fixed α) |
| α-scale | 1 / 0.25 / 0.0625 (fixed cap 20 000) |

```bash
PYTHONPATH=. python -m robot_nav.eval_gaz14_baselines --algos astar \
    --episodes 10 --seed 2000 --max-transitions 20000 \
    --log-traces runs/value_corpus/pilot/cap20k \
    --out runs/value_corpus/pilot/cap20k
```

Measure per cell: solved-plan fraction of planning calls, episode success,
**on-path value labels/ep**, expanded nodes/ep (hinge+feasibility pool),
plan_cost on episodes solved by every cell (label-quality probe), min/median
plan_cost across α-scales on matched seeds, wall-clock/ep, shard MB/ep, peak
RAM/worker.  Plus 2 corridor episodes (`--corridor`) as a world smoke test.

**Decision rules:**

* pick the **cap** where labeled-state coverage per compute-hour flattens
  (expected: 20k default is close; 50k only if solve-fraction gain is large —
  it converts censored hard states into labels, which is the point);
* pick **α-scale** = 1 unless a smaller scale measurably lowers matched-seed
  plan costs at an affordable solve-rate hit — if it does, collect at the
  smaller scale (better labels) and set the hinge slack λ from the observed
  cost ratio;
* fix worker count from RAM (31 GB box, ~1.5–2 GB + CUDA context per worker
  → 6 workers is the ceiling assumption to verify).

## 2.1 Pilot results (2026-08-26, sweep complete — decisions locked)

10 matched episodes/cell, seeds 2000–2009, scattered world, 11 workers,
~5 h wall-clock.  Results in `runs/value_corpus/pilot/`:

| cell | success | trans/ep | on-path labels | labels/100k trans |
|---|---|---|---|---|
| cap5k, α=1 | 80 % (2 collisions) | 27k | 84 | **30.6** |
| cap20k, α=1 | 90 % | 69k | 105 | 15.2 |
| cap50k, α=1 | 90 % | 149k | 90 | 6.0 |
| cap20k, α=0.25 | **100 %** | 259k | 70 | 2.7 |
| cap20k, α=0.0625 | 90 % | 305k | 34 | 1.1 |

* **Cap axis flat**: solved-plan rate 15/22/26 % across a 10× cap range;
  solved path lengths ≈ equal (9/12/10) and deepest label identical (2326).
  Searches solve when the residual problem is small, whatever the budget —
  extra cap is nearly pure waste.  **Corpus cap: 5000.**
* **α axis non-monotone**: α=0.25 → 10/10 success, 0 collisions, best mean
  cost — search effort substitutes for guidance quality (paper point).
  α=0.0625 overshoots: h stops steering (coarse share collapses to ~0),
  costs blow up (10026 on seed 2002 vs ~4430).  But low α solves only
  shallow endgame plans → terrible *label* yield.  **Corpus α: 1.**
* **λ estimate**: per-seed best-known cost / cap5k cost = **mean 0.74,
  min 0.51** (n=8) → α=1 labels carry ~26 % mean slack.  **Hinge λ = 0.7**
  (0.5 for a conservative variant).
* **Revised label economics**: exact on-path labels ≈ **8/ep** at cap5k
  (solved plans are short) — far below the old 30–80/ep guess.  500–1000 eps
  → only 4–8k exact labels; the corpus therefore leans harder on
  episode-backfill (~38 labels/ep at 80 % success), per-robot auxiliaries,
  expanded-node hinge rows (~6k/ep) and sibling resolution.  Optional:
  an α=0.25 slice purely as a backfill source (its 100 %-success episodes
  give tighter realized-cost labels, at 10× compute).
* Throughput at cap5k ≈ 12 min/ep/worker ⇒ 1000 eps ≈ 18 h on 11 workers.

## 3. Main corpus (~1 day on 11 workers)

Target: **500–1000 solved episodes → 30–80k on-path value labels**, plus
~10⁵–10⁶ expanded-node states and coarse-vet clearance labels.

* Teacher: A\*, analytic leaf at the pilot's (cap, α-scale).
* Seeds: blocks from 2000 up (episode k = seed+k; eval seeds 1000–1099 stay
  untouched).  Worlds: ~70% scattered, ~30% corridor (separate invocations,
  separate seed blocks, `--corridor`).
* 6 workers × disjoint seed blocks, shared flags, `--out` + `--merge` for the
  outcome table; `--log-traces runs/value_corpus/astar14/traces` (shards
  auto-namespace per `<algo>_<config>_s<seed>`, so workers don't collide).
* Disk: few hundred KB/plan × ~5–8 plans/ep → ~2–5 GB at cap 20k; ~10× at
  cap 50k — check free disk before choosing 50k.

**Companion sub-corpora (tagged by source in the dataset builder):**

* **PHS hard slice**: after the merge, list episodes A\* failed; rerun exactly
  those seeds with `--algos phs` (same caps).  PHS labels carry larger
  suboptimality slack (depth-penalized search) → down-weight in regression
  (§4.1) or use ranking-only.  This is the coverage of the hard tail
  (PHS 89% vs A\* 85%).
* **Episode backfill**: realized remaining cost at each decision of
  successful episodes (join per-episode records to traces by seed +
  decision_index) — outcome-grade labels on exactly the cap-hit states solved
  plans censor.  Low weight (§4.5).

## 4. Training targets (per data source)

Unit convention: all cost targets divided by `precise_unit ×
selection_interval` (= 57.85) — "precise-decision equivalents", the scale
`PerRobotValue` already used; scale back at deploy from the live cost table.

### 4.1 Value regression (primary)

On-path nodes of solved plans: `y_V(n) = plan_cost − g(n)` (exact in-model
remaining cost of the teacher's plan).  Huber loss.  Source weights:
A\* = 1.0; PHS = 0.3 (or excluded from regression, kept for ranking);
sibling-resolution children (§5) = 2.0 (the only exact off-path labels).

### 4.2 Sibling-ranking auxiliary (what Gumbel consumes)

At every on-path node: pairwise logistic/margin loss, chosen child ≺ each
sibling that is legal (`br_safe`, not collision/dead-end), margin in
normalized units.  Weak supervision (correct only insofar as the teacher
chose well) — but *more robust to the α finding than regression is*: siblings
of one node largely share the teacher's downstream slack, so ranking pairs
survive label suboptimality that distorts absolute targets.  Weight the
ranking loss up relative to the pre-α-finding instinct.  At §5 nodes: full pairwise ranking from exact
values — the high-quality slice, also the held-out **ranking gate** set.

### 4.3 Closed-set hinge (soft lower bounds — the anti-ShortCircuit data)

Every expanded (and generated) node of a *solved* search: penalize
`V(n) < λ·(C* − g(n))`, one-sided hinge.  λ < 1 absorbs the teacher's
suboptimality slack (α inadmissible → C\* not optimal); initialize λ ≈ 0.5
and raise toward the pilot's measured cost ratio if the α-sweep shows plans
are near-converged.  Small loss weight — this is a regularizer that teaches
"far states are expensive", not a calibration source.

### 4.4 Hard negatives / plentiful heads

* `br_child_collision` → collision/terminal target (feed the value's
  terminal handling or a small risk head — decide at net design; do **not**
  put terminal children in the ranking pairs).
* Every coarse vet (`br_clearance`, on- or off-path, solved or not) →
  feasibility regression `clearance − d_safe` (millions of labels; the one
  head unsolved searches still feed).

### 4.5 Backfill (outcome labels)

Realized remaining episode cost at decision points of successful episodes:
noisy, policy-dependent (AlphaGo-outcome analogue) → weight ~0.1–0.3,
and only where no trace label exists (cap-hit roots).

### 4.6 Prior targets (side-product, trained on the shared trunk)

Round 0: CE on the on-path chosen branch (`on_path_rows` aidx), over all 23
stubs, **no legality masking** (matches deployed ranking-before-vetting).
Round 0 must be one-hot: advantage-weighted soft targets would use
`br_child_g + br_child_h` with h the ~60×-inflated analytic — the weights
inherit the inadmissibility.  Round 1+: relabel with the trained value —
soft targets ∝ exp(−(step_cost + V(child))/τ) over logged children.

### 4.7 Splits and offline gates

Split **by episode**, stratified by world (never straddle a plan).  Gates:

* **G2-ranking (headline)**: sibling Spearman vs the analytic-h implied
  ranking, on held-out episodes AND on the §5 exact-sibling set — the net
  must beat analytic on both;
* value MAE/bias in normalized units, reported per world and per
  n_unreached bucket (bias vs state features is exactly the old failure);
* prior top-1/KL vs cycle_05 on held-out;
* only then closed-loop: A\*+new-h (G3: ≥ analytic's 89% at fewer than its
  35k transitions/ep — beat both bars at once), then GAZ (G4, incl. the 8
  persistent seeds).

## 5. Sibling resolution (subsample; runs while training starts)

The only source of **exact** off-path values: for a stratified subsample of
on-path nodes (~2–5/solved episode, stratified by depth × n_unreached),
reconstruct each of the 23 children (replay the branch action through
`ForwardModel14` from the stored parent poses/last_actions — deterministic),
and run an independent A\* from each child to convergence at the main cap.
Store per node: 23 × (solved, plan_cost) → exact sibling values.

Needs a small new script (`collect_sibling_values.py`: read trace shards +
meta, rebuild model per plan constants, loop).  Cost ≈ 23 × a main-corpus
search per resolved node — budget from pilot wall-clock; ~1–2k resolved
nodes is the target for the ranking gate + high-weight training slice.

## 6. Round 1+ (after first net)

* Teacher rerun with learned h (searches solve more/harder) → new labels on
  all data; keep source tags.
* DAgger-style patch: relabel states the *deployed GAZ* visits (fresh A\*
  searches from those states) if G4 failures concentrate off the teacher's
  state distribution.
