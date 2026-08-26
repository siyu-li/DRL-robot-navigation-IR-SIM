# Value / Prior Redesign: size- and configuration-generic guidance for the coupled swarm

> **Scope note (2026-08-25):** the active focus has narrowed to learning a
> better value (+prior) from A*/PHS traces for the **original** system
> (precise-all action, legacy uncoupled precise rotation).  §2 (coupled
> rotation) and §3 (precise-group configs) are implemented but **optional and
> off by default** — extensions to revisit later, with their G1 measurements
> recorded in project memory.  The network design below is a menu for the
> fresh, simpler redesign discussion, not a commitment.

Status: design, 2026-08-24.  Follows the diagnosis of the GAZ14 budget sweep
(success saturates at 77% while PHS reaches 89% with the same model — guidance,
not budget, is the bottleneck) and the rendered failure analysis (arrived-robot
jiggle; robot-robot blocking).  This document specifies:

1. the **physics fix** — coupled rotation for precise groups (§2);
2. the **generalized action space** — precise groups of size 1 or 2, generated
   in parallel (§3);
3. the **input representation** for the value and prior networks — variable N,
   variable group structure, explicit coupled-dynamics and blocking features (§4);
4. the **network architecture** (§5);
5. the **data collection plan** — supervision from A*/PHS search traces (§6);
6. the **training plan and evaluation gates** (§7);
7. phased execution order and risks (§8).

---

## 1. Design constraints

* **Size-generic.**  Must work for N = 6, 14, and later 30–40 (m = 5, 6 original
  groups).  No parameter, feature, or normalization may depend on a fixed N,
  fixed number of move-groups, or fixed group sizes (the current
  `features.py` size one-hot over {3, 4, 7} violates this and is removed).
* **Configuration-generic.**  The same network must score action sets drawn
  from different group configurations: coarse move-groups of any size,
  precise groups of 2, precise groups of 1, precise-all.  Action count is
  variable; the prior is a distribution over whatever tokens are present.
* **Dynamics-aware.**  Robot headings are *not* independently controllable:
  `rank(A) = K ≪ N` (5 for 14 robots).  Measured on `A_FULL`:
  `P = A·pinv(A)` has diagonal ≈ 0.32–0.38, i.e. a robot being steered
  jointly with everyone else realizes only ~35% of its own desired turn; and
  exactly targeting one robot's heading side-rotates bystanders by ~0.6 rad
  (mean) per rad of target.  The value cannot price a state without seeing
  this structure, so it enters the input explicitly (§4.2).
* **Blind spots covered explicitly** (from the rendered failures):
  (a) per-group composition — how many members already arrived, what progress
  the group can still buy; (b) inter-robot blocking along robot→goal rays,
  including *arrived* robots parked on another robot's ray.
* **Expansion-free prior, cheap value.**  The prior ranks all edges of a node
  from one forward pass with no transitions materialized (same contract as
  `LearnedPrior`).  The value is evaluated at every generated child — children
  of one expansion are batched into a single forward.

---

## 2. Physics fix: coupled rotation for precise control

### 2.1 What is wrong today

`SwitcherEnv._run_precise` and `ForwardModel14.precise_next` drive one robot at
a time with its GAT action while all others hold still — including their
*headings*.  Physically, any rotation is realized through the actuation matrix:
rotating robot i requires driving columns of `A`, which rotates every robot
with a nonzero entry in those columns.  Independent single-robot rotation does
not exist in this system.

### 2.2 The unified group primitive

Every action — coarse or precise — becomes one instance of the same primitive,
parameterized by a **target set** T (whose desired heading changes the solve
tries to meet), a **move set** S (who translates), and a translation length:

| | coarse(g) | precise(S) |
|---|---|---|
| target set T | all robots (LS) / members (nonlinear) | S |
| desired dθ for T | heading-to-goal errors | GT policy's angular commands |
| rotation solve | `t* = pinv(A) @ dθ_desired` (or nonlinear) | `t* = pinv(A_T) @ dθ_T` |
| applied rotation | `A @ t*` — **everyone rotates** | `A @ t*` — **everyone rotates** |
| move set S | group members | S (= T) |
| translation | fixed `move_distances[g]` | GT policy's linear commands |

`pinv(A_T)` is the minimum-norm exact solution when `rank(A_T) = |T|`.
Verified on `A_FULL`: every single row and **every pair of rows** has full
rank, so precise groups of size 1 and 2 achieve their targets exactly; the
price is the bystander side-rotation `A @ pinv(A_T) @ dθ_T` (~0.6–0.7 rad
mean per rad of target — large, and now a real modeling term).

Implementation notes:

* Per-sub-step application: at each of the `selection_interval` sub-steps, the
  GT policy's angular commands `ω_S` for the driven robots map to
  `t = pinv(A_S) @ (ω_S · dt) / dt`; all robots receive angular velocity
  `A @ t`; only S receives its linear commands.  The GT policy stays
  closed-loop (it re-observes each sub-step), so it can correct its own
  heading against incoming side-rotations on later decisions.
* Safety is unchanged by the fix: rotation in place cannot collide (circular
  footprint), and translation is still restricted to S.  The shield contract
  (coarse vetted, precise unvetted) carries over.
* Both `SwitcherEnv` and `ForwardModel14` change identically;
  `check_mpc_model` is the fidelity gate after the change.
* **All existing baselines, checkpoints, and eval numbers become
  incomparable** (same situation as the 2026-08-13 reset-distribution
  change).  Everything in §7 is measured fresh after this lands.

### 2.3 Consequence for arrived robots

Under coupled rotation, arrived robots are side-rotated by every action —
heading churn at the goal is now physically unavoidable, only *translation* of
arrived robots is avoidable.  The value must therefore price "arrived member
of a selected coarse group gets translated off its goal" (the jiggle) and
"arrived robot parked on someone's ray" (blocking) — see §4.  Note the current
action set contains **no primitive that can deliberately move an arrived robot
out of another robot's way** (the GT policy holds arrived robots at their
goals); if the learned value confirms blocking states are expensive and
unresolvable, a "yield" primitive is a candidate future action — out of scope
here, but the representation should let us measure it.

---

## 3. Generalized action space and parallel edge generation

### 3.1 Configurations to support and compare

| config | precise edges | total edges (14 robots) | sub-steps per precise edge |
|---|---|---|---|
| A: precise-all (today) | 1 | 22 + 1 = 23 | N_unreached × 5 (≈ 70) |
| B: 2-robot groups | 7 (fixed partition) | 22 + 7 = 29 | 5 (both driven simultaneously) |
| C: 1-robot groups | 14 | 22 + 14 = 36 | 5 |

* Config B pairing: use size-2 subgroups from the group algebra where a fixed
  partition exists, else a fixed index partition {(0,1), (2,3), …}.  (The
  algebra natively generates 24 size-2 and 14 size-1 subgroups for N = 14, so
  precise groups are *members of the same family* as coarse move-groups —
  keep this framing for the paper.)  Dynamic pairing (all 91 pairs) is future
  work.
* Precise edges for arrived-only groups are dropped from the node's action set
  (nothing would move); precise-all keeps its skip-arrived semantics.
* In configs B/C a "decision" is much smaller than today.  The cost table
  needs no change (`precise_unit × |S driven| × selection_interval` already
  scales), but `max_decisions` per episode must be re-derived from sub-step
  budget, not decision count, to keep episodes comparable across configs.

### 3.2 Batched precise-edge generation

Today one precise-all transition costs ~70 *sequential* GAT forwards.  With
per-group precise edges the search wants P sibling transitions, but they
batch:

* All P hypothetical child states start from the same node state.  Stack them
  into a batch of P states; per sub-step run **one batched GAT forward**
  (the GAT already computes all robots' actions per state) and one batched
  coupled-rotation integrator step.  Total: `selection_interval` batched
  forwards to materialize *all* precise children of a node — versus
  14 × 5 = 70 sequential forwards for one precise-all edge today.
* The same batching serves data collection (§6): best-first expansions
  materialize every child anyway, so this is where the wall-clock win lands.
* Budget accounting: count each materialized precise edge as one transition
  (as today); report batch-corrected wall-clock separately so effort
  comparisons across configs stay honest.

---

## 4. Input representation

All features are **relative** (no absolute px/py/gx/gy — translation
invariance and cross-world generalization), **normalized** by explicit scales
(`dist_scale` ≈ arena diameter, `clearance_cap`, π), and **per-entity**
(robot tokens, action tokens, one global token) so N and the action count are
free.  Everything below is numpy-computable from `ModelState` + per-decision
constants (goals, `ShieldGeometry`, `A`, move-group table) — sim-free and
expansion-free.  One precomputation per system: `pinv(A)`, `P = A @ pinv(A)`.

### 4.1 Per-robot token features

Geometry and dynamics state (mostly already available):

1. `d_i / dist_scale` — goal distance; plus `arrived_i` flag.
2. `cos e_i, sin e_i` — heading-to-goal error.
3. last applied action `(lin, ang)` (normalized) — dynamics state fed to the GAT.

Coupled-controllability (new — the rank-deficiency made visible):

4. **heading-correction residual** `|((I − P) dθ_des)_i| / π` where
   `dθ_des` is the vector of all robots' wrapped heading errors — the part of
   robot i's needed turn that joint least-squares steering *cannot* reach.
5. **achievable-turn fraction** `P_ii` and the achieved correction
   `(P dθ_des)_i / π` — how much of my turn the swarm-level solve delivers.

Obstacle context (as today, from `ShieldGeometry`):

6. static point clearance (capped).
7. ahead-cone clearance (obstacles within ±60° of goal bearing).

Blocking (new — blind spot (b)):

8. **robot-on-my-ray clearance**: min over j ≠ i of the corridor clearance of
   robot j to the segment `p_i → goal_i` (perpendicular distance − 2ρ, only
   for j projecting onto the segment), capped.
9. flags/aggregates of the blocker: is the nearest blocker *arrived* (a
   parked robot is a permanent obstacle under the current action set); number
   of robots inside the corridor.
10. obstacle-on-my-ray clearance (same corridor test against obstacle discs —
    sharper than the ±60° cone).
11. local crowding: mean distance to the k = 3 nearest robots / `dist_scale`.

### 4.2 Pairwise edge features (relational encoder attention bias)

For robot pair (i, j): distance `d_ij / dist_scale`; bearing of j in i's
goal-frame (cos/sin); j's arrived flag; corridor/on-ray flag from §4.1(8);
and the **coupling coefficient `P_ij`** — how much a turn desired by j bleeds
into i.  `P` depends only on `A`, so this single channel carries the entire
group-structure coupling for any N and any group algebra, and is what lets
the same network transfer across swarm sizes and configurations.

### 4.3 Per-action token features

Every candidate edge is one token: type embedding (coarse / precise) plus:

Composition (blind spot (a)):

1. `|S| / N`, `log |S|` (replaces the size one-hot).
2. **fraction of members arrived**; count of arrived members that currently
   block another robot's ray (jiggle-and-block risk in one number).
3. pooled member features (via membership-masked attention over robot tokens).

Rotation preview (new — expansion-free).  **Validated by
`robot_nav/check_coupling_features.py` (2026-08-24), which falsified the
original one-matvec spec for the deployed primitive**, so the preview is
tiered by which primitive is in use:

* `method="least_squares"` coarse, and precise groups: the linear solve
  **is** the primitive — `Δθ̂ = wrap(A @ pinv(A) @ dθ_des)` (resp.
  `A @ pinv(A_S) @ dθ_S`) matches execution to 1e-15.  One matvec.
* `method="nonlinear"` coarse (what the evals deploy): no linear formula
  reproduces the BFGS solve (multi-modal objective, path-dependent optimum).
  Measured on random states: the solver reaches ~86% of ideal member
  progress, so cross-group progress ranking is dominated by group size;
  within a size class the joint-LS preview is uninformative (Spearman
  ≈ 0.0–0.08) and the member-targeted `pinv(A_m)` preview only partial
  (0.04 / 0.43 / 0.14 for sizes 3/4/7 — size-3 groups have rank(A_m)=3, so
  the linear preview collapses to "perfect steering" and can't discriminate).
  **Therefore run the exact rotation solve per group token** — the lean-BFGS
  solve *without* frames or the swept-clearance vet — measured at
  ~0.37 ms/group ⇒ ~8 ms per node for all 22 groups.  This is exact by
  construction, still expansion-free (the vet cost the prior avoids is the
  sweep, not the solve), and adds seconds per episode against multi-minute
  episodes.

From the previewed rotation `Δθ̂` (whichever tier), derive:

4. mean/max member post-rotation heading error `|wrap(e_i − Δθ̂_i)|/π` — what
   the rank deficiency leaves uncorrected for the movers.
5. mean/max **bystander** `|Δθ̂_j|/π`, and max over *arrived* bystanders —
   the side-rotation churn this action inflicts.
6. **progress preview**: `Σ_{i∈S} (d_i − ‖p_i + move·u(θ_i + Δθ̂_i) − g_i‖)`,
   normalized by the action's step cost — the myopic efficiency ratio that
   dominated `HeuristicPrior`, now computed from the *achievable* rotation
   rather than assumed-perfect steering.  (For precise edges, the GT policy's
   direction is not known without a forward pass; use the goal-direction
   proxy.  An optional flag can spend one batched GAT forward per node to
   replace the proxy with the true commanded actions — ablation, off by
   default.)

Safety-relevant (as today, vet-free):

7. min member static clearance; min member ahead-cone clearance.
8. step cost (from `SwitcherCost` — exact, known without vetting).

### 4.4 Global token

Fraction unreached, mean/max goal distance, min static clearance, `log N`,
`rank(A)/N` (controllability richness — varies across swarm sizes), count of
arrived robots blocking someone / N (global blocking pressure).

---

## 5. Network architecture

One shared trunk, three heads (prior, feasibility, value) — the AlphaZero
shape, replacing both `PriorNet` and `PerRobotValue`:

```
robot tokens (N × d_r)           action tokens (K_a × d_a)      global (d_g)
      │                                  │                          │
  2 encoder layers over robots           │                          │
  (attention with edge-feature bias      │                          │
   from §4.2: geometry ⊕ P_ij ⊕ ray)     │                          │
      │────────── membership-masked ─────┤                          │
      │           + full cross-attention │                          │
      │                                  │                          │
      ├── per-robot value head v_i       ├── policy logit head      │
      │   (masked to unreached)          ├── clearance-margin head  │
      │                                  │   (coarse tokens only)   │
      └── V(s, action set) = Σ_i v_i  +  global-residual MLP(pool ⊕ global ⊕ pooled action tokens)
```

Design decisions and rationale:

* **Relational encoder, not a deep-sets MLP.**  Blocking is irreducibly
  pairwise; the edge-feature bias injects it (and the coupling `P_ij`)
  directly instead of hoping attention discovers it.  At N ≤ 40 tokens the
  transformer cost is trivial.  A deep-sets ablation (drop the encoder
  layers, keep the features) isolates how much the relational part buys.
* **Value = context-aware per-robot sum + global residual.**  The old failure
  was not the sum readout per se but summing *context-free, precise-only,
  outside-collected* per-robot values.  Here each `v_i` is computed after
  message passing (a blocked robot's token knows it is blocked) and trained
  end-to-end on joint labels (§6), with per-robot auxiliary supervision.
  The sum keeps the output scale linear in N (size generalization) and gives
  per-robot cost attribution — directly renderable for failure analysis.
  The residual term captures whatever is not attributable per robot.
* **Value is conditioned on the action set** — the value readout sees pooled
  action tokens, because cost-to-go under config B ≠ config C ≠ precise-all.
  One network then serves all configurations and can be trained on mixed
  data.  (Risk: config interference; fallback is a config embedding or
  separate value heads — ablate.)
* **Prior over variable tokens**: one logit per action token, softmax over
  whatever tokens exist — 23, 29, 36, or any future action set, unchanged.
* **Feasibility head** stays as-is conceptually (regress the shield's
  `clearance − d_safe`; threshold at deploy) — it worked (feas_acc ~86%) and
  saves vets in the lazy tree.
* **Output normalization**: train the value in "precise-decision equivalents"
  (cost / `(precise_unit × selection_interval)`), the unit `PerRobotValue`
  already uses; scale back at deploy from the live cost table.  Targets are
  additionally divided by nothing else — `Σ v_i` already scales with
  N_unreached.
* Size: ~2 layers, d ≈ 64–128, ≲ 200k params.  One forward per expanded node
  (prior + feasibility + value of the node) and one *batched* forward for all
  children of an expansion (values of children).  No frozen-GAT dependency in
  the guidance path (the geometry variant already beat the embedding variant;
  the GAT stays for generating precise actions only).

---

### 5.1 Post-G1 refinements (2026-08-25)

Driven by the stage-1 measurements (granularity 2%→50%; precise-only 67%
success / **29% collisions** under coupling) and the first A* traces
(~12% of plans solved at cap 5000 → value labels are the scarce resource;
clearance labels number in the millions):

* **Fourth head — precise collision risk**: per precise token, predict the
  probability the rollout ends predicted-in-collision (labels:
  ``br_child_collision``).  The precise analogue of the coarse margin head —
  needed now that precise edges are cheap/numerous and the GT policy collides
  more under coupled physics.  The value stays collision-free by construction
  (in-model plans censor collision children); risk heads + shield own safety.
* **Per-robot auxiliary labels are the answer to value-label scarcity**:
  arrival indices + per-step cost attribution (coarse split over members,
  precise to driven) are offline-derivable from solved paths — ~N× the
  effective supervision, and they pin the ``v_i`` decomposition.
* **Episode-backfill root labels** (collect_leaf_data-style realized remaining
  cost on successful episodes, joined to traces by seed + decision index)
  complement solved-path labels on exactly the states plans cap-hit on.
* **Prior CE over all stubs, no legality masking** — matches the deployed
  ranking-before-vetting semantic; A*-sourced actions are prior-free (clean
  cold start).
* Corpus spans configs pairs+singles → the action-set conditioning of the
  value is exercised by construction.
* Open: raise the 5000 cap for a dedicated fresh-seed collection run if
  solved-plan volume stays short (fresh seeds are needed for eval hygiene
  anyway — current traces are on eval seeds 1000–1099).

## 6. Data collection: best-first search traces as the teacher

Rationale (from the sweep): A* (85%) and PHS (89%) with the *same* model and
action set already contain the decisions GAZ's shallow lazy tree never
discovers.  Their solved plans are exact in-model trajectories with exact
remaining costs at every node — labels the shallow self-distillation loop
(val_kl flat at ~1.38, plateaued after cycle 2) could not produce.

### 6.1 Labels harvested per solved plan

Instrument `BestFirstSearch14.run` (a `trace` hook already exists; extend it
to dump on solve):

* **Value targets** — for every node on the root→goal path:
  `y_V(node) = g(goal) − g(node)` (exact in-model remaining cost of the found
  plan; an upper bound on optimal — the standard search-as-teacher target).
* **Per-robot auxiliary targets** — replay the plan through the forward
  model, record for each robot the accumulated cost share and the decision
  index at which it arrives; per-robot remaining cost supervises the `v_i`
  head.  (Attribution rule: coarse cost split over members; precise cost to
  the driven set.)
* **Prior targets** — at every on-path expanded node, the plan's chosen edge
  (cross-entropy).  Best-first materializes *all* children at expansion, so
  also log every sibling's `(g_child, h_child, step_cost)` for soft/
  advantage-weighted targets as an ablation.
* **Feasibility targets** — every coarse vet anywhere in the search (on-path
  or not) yields an exact `clearance − d_safe` label; these are plentiful and
  need no solved episode.
* **State snapshot per node**: poses + last_actions (+ episode constants once
  per shard: goals, obstacle geometry, A, move-group table, config id).
  Features are built offline from snapshots, so feature iterations never
  re-run the sim/search.

Censoring: unsolved searches (cap-hit with no generated goal node) contribute
only feasibility labels; value/prior labels come from solved plans only —
same censoring discipline as `collect_leaf_data.py`.

### 6.2 Collection mechanics

* Teachers: A* (cheapest per episode, 12k transitions/ep) for volume; PHS
  (43k/ep, strongest) for a smaller high-quality slice.  Both via
  `eval_gaz14_baselines.py` infrastructure (workers, shards, `--merge`).
* **Seeds disjoint from eval**: eval stays 1000–1099; train seeds e.g.
  2000–2999.  Worlds: scattered + corridor.  Configs: A, B, C mixed (tag each
  shard) — mixed-config training is what buys configuration generality.
  Optionally add the 6-robot system for cross-N training (else hold it out
  as the zero-shot test).
* Volume: plan lengths are ~30–80 decisions; ~500–1000 solved episodes →
  ~30–80k value/prior samples + orders more feasibility labels.  Throughput
  estimate from the sweep (GAZ b100 ≈ 5.3k transitions ≈ 5–8 min/ep,
  single-core): A* ≈ 10–20 min/ep ⇒ with 6 workers ≈ 1–1.5 days per 500
  episodes.  **Run a 10-episode pilot first to measure actual throughput and
  shard size before committing.**  Mind the 31 GB RAM box: ~1.5 GB/worker is
  fine, but batched child generation on CUDA adds a context per worker.
* Round 0 uses the current `value_geometry` + cycle_05 prior to guide the
  teacher searches (best known h, π); later rounds use the new net (§7).

## 7. Training and evaluation

### 7.1 Training

* Joint multi-task loss:
  `L = L_V (Huber, normalized units) + λ_pr L_prior (CE/KL) + λ_f L_feas
  (Huber on margin) + λ_aux L_perrobot`.  Start λ ≈ 1, tune by validation.
* Split **by episode** (never straddle), stratified by config and world.
* Offline metrics that matter (not just MAE):
  - value **sibling ranking**: Spearman correlation between predicted and
    realized `y_V` across children of the same node — ranking children is
    the job, calibration second;
  - prior top-1 / KL vs plan actions; feasibility precision/recall at the
    deploy threshold;
  - per-config and per-N breakdowns (the generality claim is a table, not an
    average).

### 7.2 Iteration loop (search-as-teacher policy iteration)

1. Round 0: collect with current h, π → train net v1.
2. Re-run teachers with v1 (searches get cheaper and solve more of the hard
   seeds → better and broader traces) → train v2 on all data.
3. Stop when teacher success/effort and offline metrics plateau (expect 1–2
   rounds; the old loop's plateau was a data-quality ceiling, not a compute
   ceiling — if v1 ranking metrics don't beat the analytic h's implied
   ranking, stop and re-diagnose features before collecting more).

### 7.3 Evaluation gates (all measured after the §2 physics fix re-baseline)

| gate | criterion |
|---|---|
| G0 model fidelity | `check_mpc_model` passes with coupled precise rotation |
| G1 re-baseline | precise-only / coarse-only / A* / PHS / GAZ re-run under new physics, configs A/B/C, analytic h — the new comparison floor (also a paper table: precise granularity at fixed guidance) |
| G2 offline | v1 beats analytic h on sibling ranking; prior beats cycle_05 on held-out KL/top-1 |
| G3 teacher | A* + new h: ≥ its G1 success at fewer transitions/ep |
| G4 target | GAZ-L b100/200 + new h,π: success → toward PHS-level (was 77% vs 89%), improvement on the 8 persistent seeds (1012 1031 1051 1066 1073 1085 1088 1095), coarse breaches stay 0 |
| G5 generality | zero-shot 6-robot eval; config-transfer eval (train mixed A/B/C, test each); stretch: 30-robot m=5 world |

## 8. Execution order and risks

Phases (each lands independently):

0. **Physics fix** (§2): unified primitive in `SwitcherEnv` + `ForwardModel14`
   (+ 6-robot variant for cross-N runs), `check_mpc_model` gate, re-baseline.
   *Do this first — any data collected under the wrong dynamics is wasted.*
1. **Action space** (§3): per-group precise edges in stubs/search, batched
   child generation, config A/B/C flag through the eval scripts. G1 table.
2. **Features + net** (§4, §5): N-generic feature builder (replaces
   `features.py`'s fixed-N assumptions), encoder + 3 heads, checkpoint format.
3. **Collection** (§6): trace instrumentation, collector CLI, 10-ep pilot,
   then the round-0 corpus.
4. **Training** (§7.1): trainer + offline metric report (extends
   `train_prior.py` / `train_value.py` patterns).
5. **Closed-loop + iteration** (§7.2–7.3): wire into A*/PHS/GAZ, gates G2–G5.

Risks / open questions:

* **GT policy under coupled rotation** — the frozen GAT was trained without
  incoming side-rotations; closed-loop it should partially compensate, but
  precise-only success may drop at G1.  If it drops badly, a short finetune
  of the GAT under the new dynamics becomes a prerequisite (known playbook:
  `marl_finetune_partial_inactive`).
* **In-model labels vs sim drift** — labels are in-model costs; GAZ executes
  closed-loop so drift is bounded per decision, and the guidance only ranks.
  Keep the G4 closed-loop gate as the arbiter.
* **Value conditioning on action set** — mixed-config training could
  interfere; fallback per §5.
* **2-robot pairing choice** is a free design variable; fixed partition
  first, note as limitation.
* **Arrived-robot blocking has no resolving action** — the value will learn
  to see it; if G4 failures concentrate there, a yield primitive is the next
  action-space extension (paper discussion point).
