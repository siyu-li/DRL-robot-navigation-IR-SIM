# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & commands

Dependencies are managed with **Poetry** (`pyproject.toml`, Python ≥3.10); the
active local conda env is `DRL_robot_nav`. All entry points are Python modules,
run from the repo root with `python -m robot_nav.<script>`.

```bash
poetry install                                  # install deps (includes ir-sim[all], torch, torch-geometric)
poetry run pytest                               # run tests (CI runs this on master; testpaths=tests)
poetry run pytest tests/inpect_model.py         # single test file
tensorboard --logdir runs                       # training curves

# MARL TD3 with obstacle graph nodes (the frozen "precise" backbone)
python -m robot_nav.marl_train_obstacle_6robots      # also _14robots variants
python -m robot_nav.marl_test_obstacle_6robots       # also marl_test_coarse_6robots
python -m robot_nav.marl_finetune_partial_inactive   # anneals p_inactive 0→70% for group-switching

# CAPSwitcher tree-search eval / fidelity check (see "GPU box" note below)
python -m robot_nav.eval_mpc --episodes 100 --depths 1 2 3 --baselines
python -m robot_nav.eval_mpc --algos mpc mcts gumbel --budgets 5 21 \
    --value-model <ckpt> --log-pi-targets data/pi_targets   # budget-matched comparison
python -m robot_nav.check_mpc_model --episodes 5

# CAPSwitcher learned value (cost-to-go) pipeline
python -m robot_nav.collect_value_data           # MC rollouts → data shards
python -m robot_nav.train_value --data data/value_data \
    --out-dir robot_nav/models/MARL/capswitcher/checkpoint/value   # sim-free, runs locally
```

**GPU box gotcha:** the locally-installed `irsim` crashes inside `sim.step`
(a `step_status` error), which breaks anything that actually advances the
simulator — training, MARL tests, `eval_mpc`, `check_mpc_model`. Run those on
the GPU box. Sim-free work (`train_value`, unit tests, static analysis of saved
shards) runs fine locally. Several script docstrings repeat this warning.

Trained weights live under `.../checkpoint/` directories that are **not checked
into git** (e.g. the frozen GAT backbone at
`robot_nav/models/MARL/marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive/`
referenced by `eval_mpc.build_env`). Expect checkpoint paths in code to be
absent locally.

## Architecture

Two largely independent bodies of work share the `robot_nav/` tree.

### 1. Single-agent + MARL TD3 navigation (original repo)

Classic DRL navigation in IR-SIM: a robot uses 2D laser + goal info to learn to
reach a point. `robot_nav/models/` holds interchangeable algorithms (TD3, SAC,
PPO) documented in the README; the single-agent driver script the README names
(`rl_train.py`) is not present in this checkout, so the live entry points are the
MARL scripts below. `SIM_ENV/sim_env.py`
wraps IR-SIM for the single-robot case; `SIM_ENV/marl_obstacle_sim.py`
(`MARL_SIM_OBSTACLE`) extends it to multi-robot with **obstacle graph nodes** and
Shapely-based clearance. The MARL policy is a **graph-attention TD3**
(`models/MARL/marlTD3/marlTD3_obstacle.py` → `ActorObstacle`, whose encoder is
`models/MARL/Attention/iga_obstacle_optimized.py::AttentionObstacleOptimized`):
per-robot 11-dim state + obstacle nodes → hard (Gumbel) proximity gates → soft
message passing → decoder → per-robot lin/ang velocity. This trained
network is **frozen** and reused as the "precise" controller below.

### 2. CAPSwitcher (`models/MARL/capswitcher/`, active work on branch `CAPSwitcher`)

A switcher for a coupled unicycle swarm (6 robots, 3 coarse groups) that at each
decision chooses between two control regimes:

- **Coarse** (`policies/coarse_steering.py`): activating a group rotates all
  robots via a **rank-deficient actuation matrix** (artificially rank-2), then
  only that group's members translate forward by a fixed `move_distance`.
  Physically efficient, but cannot independently steer or avoid obstacles.
  Solved either by least-squares or nonlinear (BFGS) over the reduced matrix.
- **Precise** (`policies/gat_backbone.py::GATBackbone`): the frozen MARL GAT run
  **one robot at a time** (each robot driven for `selection_interval` sub-steps
  while others hold still). Obstacle-aware but physically wasteful.

The switcher's objective is to **minimize precise usage / physical path cost
subject to safety**. `capswitcher_project_summary.md` in that directory is the
authoritative design doc (physics, actuation matrix, GAT cut-point analysis);
read it before touching this subtree. The current research direction
**decouples** the problem into two orthogonal pieces:

- **Safety = local & exact.** `rl/shield.py` (`ShieldGeometry`) reconstructs the
  exact swept geometry of a coarse move from its frames and vets clearance
  `>= d_safe` at every sub-step *before* committing — so unsafe coarse moves are
  never executed. This is local and scales to large swarms.
- **Efficiency = global sequential planning.** `rl/forward_model.py` is the
  deterministic analytic pose-model every planner searches over, because IR-SIM
  cannot be branched (`ModelState`, unicycle forward-Euler for precise, exact
  reconstruction for coarse; `cost_to_go` is the search leaf;
  `build_forward_model` rebuilds it from the live sim each real decision).
  All planners live in `rl/search/`, share that model, the expansion machinery
  (`common.py::expand` — shield-safe coarse groups ∪ precise-all), leaf value and
  the `decide(robot_state) → {mode, group, frames, candidates}` dict contract;
  the resource is **node expansions per decision** (measured by
  `ForwardModel.n_precise_expansions`; exhaustive depth-d ≤ (4^d−1)/3).
  Layering is strictly one-directional:
  `reward/shield → search/common → forward_model → tree/minimin → mcts/gumbel`
  (keep it that way — `common.py` must stay a dependency leaf):
  - `minimin.py` — exhaustive fixed-depth min-cost recursion (`plan_decision`,
    the MPC baseline) + `MPCSwitcher`, the receding-horizon switcher and base
    class of the budgeted switchers.
  - `tree.py` — shared Node/backup: **Bellman value replacement** default
    (`"mean"` ablation), **certificates** `U` (exact in-model cost of a found
    complete plan) stored separately from estimates `q̂`, effective value
    `min(q̂, U)`, positive-cost branch-and-bound prune vs the root incumbent.
  - `mcts.py` — UCT MCTS (`MCTSSwitcher`), learned value as leaf, no rollouts.
  - `gumbel.py` — **Gumbel AlphaZero** planning (`GumbelSwitcher`): Sequential
    Halving root + deterministic non-root rule (Danihelka et al. 2022); emits
    the improved root policy `pi_prime` per decision (future prior-net targets).
  - `priors.py` — `HeuristicPrior`, expansion-free base policy from shield
    stats (progress/step_cost ratio, clearance margin, precise bias).
  Sim-free tests: `tests/test_tree_search.py` (stub-MDP equivalence vs minimin).

`rl/switcher_env.py` (`SwitcherEnv`) is the gym-like wrapper tying sim + frozen
backbone + coarse primitive together; observation is per-robot 512-d GAT decoder
embeddings kept **unpooled** (a Deep Sets head, `policies/deep_sets_head.py`,
learns its own permutation-invariant pooling). `rl/reward.py::PathCostReward`
charges each decision its executed motion cost (flat `precise_cost` for precise,
`n_members · move_distance` for coarse).

**Value learning (in progress).** The crude analytic leaf `ĥ = α·Σ‖p−goal‖` is
being replaced by a learned per-robot precise cost-to-go `v_ψ`:
`collect_value_data.py` generates MC labels from 6-robot rollouts (3 robots held
static as neighbors, 3 driven by the precise policy), `train_value.py` fits the
regressor (`rl/value_net.py`), and `LearnedCostToGo` plugs into
`ForwardModel.cost_to_go` as a selectable leaf (`eval_mpc --value-model`, rows
`MPC-d*+v`). Two feature variants share one checkpoint format: `"embedding"`
(512-d GAT embedding, neighbor-aware) and `"geometry"` (raw 11-col state,
congestion-blind baseline). Longer-term plan: budget-matched MPC/MCTS/Gumbel
comparison on the GPU box → learned Deep-Sets prior distilled from the logged
`pi_prime` targets (`eval_mpc --log-pi-targets`) → full AlphaZero training loop.

### Key invariants when working on CAPSwitcher

- Coarse decisions must be **exactly reconstructable** from their frames — the
  shield's safety guarantee and the MPC model's coarse fidelity both depend on
  the executed control equaling the vetted/scored control. `check_mpc_model.py`
  asserts coarse matches the sim *exactly* and precise *within tolerance*.
- The GAT backbone is frozen everywhere it is used; never train through it.
- Per-robot embeddings are deliberately left unpooled through the env — pooling
  is a learned decision, not an env-level one.

  
