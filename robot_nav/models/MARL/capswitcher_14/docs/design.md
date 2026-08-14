# CAPSwitcher-14 — lazy Gumbel AlphaZero over 22 coarse groups

The 14-robot instantiation of the coarse/precise switcher, built as a sibling
package of `capswitcher` (the 6-robot system, untouched).  The physical
problem statement is unchanged — see [`capswitcher/docs/design.md`](../../capswitcher/docs/design.md)
§1–4.  This document records what is *different*: the corrected search
rationale, the lazy tree, and the 14-robot system constants.

## 1. The corrected rationale

The 6-robot searches evaluate **every child at node expansion**: all coarse
groups shield-vetted, the precise rollout run, every child leaf-evaluated.
The prior was then built *from the vetting output*, justified as "free because
vetting is mandatory anyway."  That rationale is wrong: vetting is mandatory
only for the one action **executed at the root**.  At interior nodes,
verifying a coarse action = building its frames and sweeping them = computing
its transition — verification *is* expansion.  A prior fed by vetting spends
the very budget it exists to save.

Gumbel AlphaZero's premise (Danihelka et al., ICLR 2022 — "Policy Improvement
by Planning with Gumbel") is the opposite economics: **ranking actions is
cheap** (one policy forward pass covers all edges), **evaluating them is
expensive** (one transition per edge).  At 14 robots the gap is decisive — an
eager node expansion costs 22 single-group vets + one precise rollout
(14 robots × 5 sub-steps = 70 sequential GAT forwards) + 23 leaf evaluations,
where one PriorNet forward is a tiny MLP pass.  A policy provably shrinks
search effort in exactly our setting (single-agent deterministic min-cost):
Orseau & Lelis, "Policy-Guided Heuristic Search with Guarantees", AAAI 2021.

## 2. The lazy search (`rl/search/`)

Per real decision (receding horizon):

* **Root — eager.**  All 22 groups vetted by the real shield, precise rolled
  out.  This keeps the safety contract (the executed action always carries
  exactly-vetted frames), prunes infeasible root actions before the bandit
  sees them, and harvests 22 exact clearance labels per decision for the
  feasibility head.  Cost: 23 transitions, charged to the budget honestly.
* **Below the root — lazy** (`tree.py`).  Expansion = branch stubs + one prior
  forward (cheap).  A descent buys **one** transition: the first
  unmaterialised edge it selects.  For a coarse edge the vet is the
  transition, so verify-on-descent is free relative to the cost the search
  must pay anyway.
* **Refuted coarse edge = illegal action discovered late.**  The shield
  forbidding a move is not a bad outcome to score: the edge is *pruned* from
  the node's legal set — no Q value, no collision penalty.  The precise edge
  is always legal, so the legal set can never go empty.  `COLLISION_COST`
  survives only for **states** predicted in collision (possible for precise
  children, which are not shield-vetted).
* **Completed-Q.**  Wherever a policy over edges is needed, unmaterialised
  legal edges take the paper's mixed value estimate
  `v_mix = (v_node + N·q̄_π) / (1 + N)` (prior-weighted mean of materialised
  values), in min–max-normalised space.  Evidence adjusts an edge's standing;
  absence of evidence leaves it at the prior's.
* **Bellman value replacement over materialised edges only.**  The leaf
  estimate `h` holds a node's value only while nothing is materialised — a
  low-lying `h` must never sit inside the min next to real edge values, or it
  sticks forever when its edge is soundly never bought (e.g. excluded by the
  branch-and-bound prune).
* **Root bandit** (`gumbel.py`): Gumbel-top-m over legal root edges,
  Sequential Halving by `gum + logits + σ(q)`, act with the survivor.
  Non-root: deterministic `argmax π'(a) − N(a)/(1+ΣN)` with
  `π' = softmax(logits + σ(completedQ))`, steered by the prior's *advisory*
  feasibility mask (real vets and the precise edge always override it).
* **Certificates + branch-and-bound** carry over from the 6-robot tree,
  restricted to materialised children.

**Budget = `model.n_transitions`** (single-group vets + precise rollouts,
refuted vets included).  This is the honest unit: the eager 6-robot scheme's
"budget = node expansions" hid a ~k× multiplier inside each expansion.

Known property (inherent to the family, documented in `gumbel.py`): non-root
selection is near-argmax, so a pessimistically-estimated subtree may never be
materialised and corrected; optimistic errors self-correct on contact.  Root
arms are protected by Sequential Halving.  Full-budget equivalence with
exhaustive minimin holds when leaf estimates are exact
(`tests/test_tree_search_14.py`).

## 3. Safety story

Three layers, from hard to soft:

1. **Execution**: the root action is always the exact plan the real shield
   vetted (frames run verbatim) — identical guarantee to the 6-robot system.
2. **In-tree, on contact**: any coarse edge the search actually descends is
   exactly vetted; a wrong "feasible" prediction is corrected by pruning.
3. **In-tree, at range**: the feasibility head's predicted clearance margin
   masks likely-refuted edges before they are bought.  A wrong "infeasible"
   prediction costs only efficiency, never correctness.

## 4. The 14-robot system (`configs.py`)

Binary group algebra with `m = 4` original groups; robots are the non-trivial
codes 1..14 (`policies/group_generator.py`).

* **Actuation** `A_FULL` (14×5): the four complement columns + the all-ones
  empty-group column.  rank = 5 < 14 — naturally rank-deficient, **no
  artificial column drop** (settled).  Consequences: the whole coarse control
  is deterministic given (state, group) — the 6-robot seed-threading machinery
  is gone — and `pinv(A)` is computed once.  Note the behavioural shift from
  the 6-robot system: rotation no longer depends on the chosen group; groups
  differ in *who translates* (and in the nonlinear solve's member objective).
* **Coarse actions**: all **22** subgroups of size 3/4/7 (6 + 12 + 4), sorted
  by (size, indices); list position = action id everywhere.  Group size is
  orthogonal to rank deficiency — size decides who moves, rank constrains the
  rotation subspace.
* **Precise**: the frozen 14-robot GAT, one robot at a time.  Robots already
  within `goal_threshold` are **skipped** — they neither move nor cost.

## 4b. Decision costs (`cost_14robots.yaml`)

Pricing is data, not code: `SwitcherCost` (`capswitcher/rl/cost.py`, shared by
both instantiations) loads a YAML listing, per coarse action id, the group's
`members`, its `move_distance` and its `cost`.  Costs are **free constants**,
no longer derived from `n_members · move_distance` (`cost: auto` still means
that product).  The listed members are validated against the live group
algebra at construction, so a reordering fails loudly instead of silently
mispricing.  Per-group `move_distance` also drives the primitive itself —
`CoarseSteering14` takes the whole `{group: distance}` table.

Precise is priced per robot **per sub-step**:

    precise cost = precise_unit × (robots driven) × (sub-steps each)

The planner charges the nominal `precise_unit · n_unreached(ms) ·
selection_interval` (so the precise edge gets cheaper as robots finish); the
environment charges the sub-steps actually executed.  The two differ only when
a terminal event truncates a rollout.  The leaf heuristic follows the same
unit: `α = precise_unit / (lin_max · step_time)` over *unreached* robots only.

There is no reward-shaping layer left: the environment's reward is `−path_cost`
and collision / all-reached / out-of-bounds are `done` flags, not bonuses.

## 5. Phase 0 → distill loop

1. **Phase 0 (teacher)**: `eval_gaz14_lazy --budgets 100 --log-pi-targets ...`
   with the uniform prior.  Each real decision logs: root features
   (`features.py` — static, *never* vetting output), all 22 clearance labels,
   the legal mask, and π′ (root-only distillation target — settled).
2. **Distill**: `train_prior.py` — masked KL on π′ + Huber on the clearance
   margin, one trunk, two heads.  Pass **all** iteration directories to
   `--data` (replay mixing across teachers).
3. **Deploy / iterate**: `eval_gaz14_lazy --prior-model <ckpt>` at matched or
   smaller budgets; re-collect, re-train.  Refresh the value net
   (`train_value`) in the same cycle — π′ quality is bounded by the Q it is
   computed from.

Certificates give upper bounds only; no optimality claims (settled).

## 6. File map

```
configs.py                     A_FULL, MOVE_GROUPS (22), make_coarse_steering
cost_14robots.yaml             per-group members/move_distance/cost, precise_unit
policies/group_generator.py    binary group algebra (moved from capswitcher)
policies/coarse_steering.py    CoarseSteering14: full-A rotation, no drop RNG
rl/forward_model.py            ForwardModel14: coarse_move (single edge),
                               precise_next, n_transitions
rl/search/common.py            Branch stubs, expand_stubs, QNormalizer
rl/search/features.py          GroupFeatureBuilder (static features only)
--- GAZ14-L (lazy): per-edge materialisation, completed-Q --------------------
rl/search/tree.py              lazy Node, materialize, completed_q, backup
rl/search/gumbel.py            GumbelAlphaZero14 + GumbelSwitcher14
rl/search/priors.py            UniformPrior (stub-only contract)
rl/search/prior_net.py         PriorNet + LearnedPrior (logits + feasibility)
--- GAZ14-E (eager): whole-node expansion, no completion ---------------------
rl/search/tree_eager.py        expand_node_eager, simulate_eager, expansion_cost
rl/search/gumbel_eager.py      GumbelAlphaZero14Eager + GumbelSwitcher14Eager
rl/search/priors_eager.py      HeuristicPrior14 (post-vet contract)
-----------------------------------------------------------------------------
robot_nav/eval_gaz14_lazy.py   eval + shard collection, GAZ14-L (GPU box)
robot_nav/eval_gaz14_eager.py  eval + shard collection, GAZ14-E (GPU box)
robot_nav/iterate_gaz14.py     plan->distil loop, GAZ14-L (GPU box)
robot_nav/train_prior.py       distillation training (sim-free, local OK)
tests/test_capswitcher_14_config.py   pinned group lists, A_FULL, determinism
tests/test_tree_search_14.py          lazy search semantics + minimin anchor
tests/test_tree_search_eager_14.py    eager semantics, atomic budget, prior
tests/test_prior_net_14.py            features/net shapes, prior contract
tests/test_switcher_cost.py           cost formulas, drift validation, YAMLs
```

The two search variants share `common.py` / `features.py` and the fixed
23-wide branch layout, so their logged π′ shards are interchangeable and
`train_prior.py` consumes either teacher unchanged.

Reused from `capswitcher` (imported, not forked): shield sweep geometry,
`SwitcherCost`, `GATBackbone`, `SwitcherEnv`, `LearnedCostToGo`.
