# CLAUDE.md

DRL robot navigation in IR-SIM. Two bodies of work share `robot_nav/`:

1. **Single-agent + MARL TD3 navigation** (original repo) — laser + goal → point navigation.
   The MARL policy is a graph-attention TD3 (`models/MARL/marlTD3/` + `models/MARL/Attention/`).
2. **CAPSwitcher** (`models/MARL/capswitcher/`, branch `CAPSwitcher`) — **the active research**:
   a switcher choosing between coarse group control and precise per-robot control for a coupled
   unicycle swarm. Target: a paper, and scaling beyond the current 6 robots.


## Environment & commands

Python ≥3.10. Use the conda env **`DRL_nav`**
(`/home/siyu/miniconda3/envs/DRL_nav/bin/python`) — `pyproject.toml` describes a Poetry setup,
but Poetry is not installed on this machine and plain `python` resolves to miniconda `base`,
which has no pytest. Entry points are modules run from the repo root with `PYTHONPATH=.`.

```bash
PYTHONPATH=. python -m pytest -q                 # testpaths=tests; CI runs this on master
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

- **This machine runs the simulator fine** under `DRL_nav` (irsim 2.5.5, CUDA available) —
  `env.reset()`, `env.step()` and the full 14-robot switcher loop were verified end to end on
  2026-08-17. An earlier note here claimed `sim.step` was broken locally with a `step_status`
  error and that sim work had to move to "the GPU box"; that error comes from a different
  interpreter (miniconda `base`), not from `DRL_nav`, and this *is* the GPU box.
- **31 GB RAM, 1 GB swap.** A run that dies with a bare `Killed` and no traceback was
  OOM-killed, not crashed. Confirm with `journalctl -k | grep -i oom` before debugging it as
  anything else.
- **Checkpoints are not in git.** Paths in code point at absent directories locally — e.g. the
  frozen GAT backbone `marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive/` referenced
  by `eval_mpc.build_env`. Expected, not a bug.
