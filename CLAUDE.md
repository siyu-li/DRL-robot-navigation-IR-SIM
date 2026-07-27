# CLAUDE.md

DRL robot navigation in IR-SIM. Two bodies of work share `robot_nav/`:

1. **Single-agent + MARL TD3 navigation** (original repo) — laser + goal → point navigation.
   The MARL policy is a graph-attention TD3 (`models/MARL/marlTD3/` + `models/MARL/Attention/`).
2. **CAPSwitcher** (`models/MARL/capswitcher/`, branch `CAPSwitcher`) — **the active research**:
   a switcher choosing between coarse group control and precise per-robot control for a coupled
   unicycle swarm. Target: a paper, and scaling beyond the current 6 robots.


## Environment & commands

Poetry (`pyproject.toml`, Python ≥3.10); local conda env `DRL_robot_nav`. Entry points are
modules run from the repo root.

```bash
poetry run pytest                                # testpaths=tests; CI runs this on master
tensorboard --logdir runs

python -m robot_nav.marl_train_obstacle_6robots  # also _14robots; marl_test_*; marl_finetune_partial_inactive
python -m robot_nav.check_mpc_model --episodes 5 # forward-model fidelity gate — run after touching coarse/shield/forward_model
python -m robot_nav.eval_mpc --episodes 100 --depths 1 2 3 --baselines
python -m robot_nav.eval_mpc --algos mpc mcts gumbel --budgets 5 21 \
    --value-model <ckpt> --log-pi-targets data/pi_targets
python -m robot_nav.collect_value_data           # MC rollouts → data shards
python -m robot_nav.train_value --data data/value_data --out-dir <ckpt-dir>
```

## Gotchas

- **Local irsim is broken inside `sim.step`** (`step_status` error) — this breaks *everything*
  that advances the simulator: training, MARL tests, `eval_mpc`, `check_mpc_model`. It is a
  pre-existing environment mismatch, **not** caused by new code; don't debug it as if it were.
  Run those on the GPU box. Sim-free work (`train_value`, unit tests, analysis of saved shards)
  runs locally.
- **Checkpoints are not in git.** Paths in code point at absent directories locally — e.g. the
  frozen GAT backbone `marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive/` referenced
  by `eval_mpc.build_env`. Expected, not a bug.
