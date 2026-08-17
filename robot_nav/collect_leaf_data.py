"""
Harvest leaf cost-to-go labels for the 14-robot switcher — **stage 1** of the
learned-leaf plan: measure how wrong the analytic heuristic is before building
anything to replace it.

What is measured
----------------
At every real decision the switcher makes, we log the analytic leaf heuristic
``h = α · Σ_{i unreached} ‖p_i − goal_i‖`` (via the *same*
:func:`analytic_cost_to_go` the search plans with) and, by backfilling at
episode end, the **realised remaining cost** ``G_t = Σ_{s ≥ t} step_cost_s``
under the policy that was actually running.

The quantity of interest is the ratio ``G_t / h_t``.  The heuristic prices the
whole completion at the *precise* rate (77.1 per robot-metre on the shipped
cost table) while a size-7 coarse move costs 1.26 per robot-metre — 61× less —
so ``h`` is expected to overestimate badly.  That is the direction the Gumbel
searches do **not** self-correct (pessimistic leaves persist; see
``search/gumbel_eager.py``), which is why it is worth quantifying first.

The decisive question this answers, before any network is written:
**is the ratio roughly constant?**  If it is, the fix is a corrected ``α`` — a
one-line change — not a learned leaf.  ``analyze_leaf_data.py`` reports exactly
that.

Censoring
---------
Only episodes that **reach the goal** carry valid returns: a timeout gives a
truncated sum (an under-estimate of the true remaining cost) and a collision
ends the episode early for reasons the cost does not express.  Both are dropped
from the labelled set rather than silently biasing it toward optimism — the
direction that makes the search waste budget.  ``--keep-censored`` retains them
with ``valid=False`` for inspection.

Shards also carry ``group_feats`` / ``global_feats`` per decision, so stage 2
(the value head on ``PriorNet``) trains from these files directly with no
re-collection.

Usage (GPU box — local irsim step crashes; see project memory):
    python -m robot_nav.collect_leaf_data --episodes 100 --variant eager \
        --budget 115 --out data/leaf_14e

    # the lazy variant, for comparison of the two policies' realised costs
    python -m robot_nav.collect_leaf_data --episodes 100 --variant lazy \
        --budget 115 --out data/leaf_14l

Then, locally:
    python -m robot_nav.analyze_leaf_data --data data/leaf_14e
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from robot_nav.eval_gaz14_lazy import (
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_COST_CONFIG,
    build_env,
    resolve_device,
    run,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    analytic_alpha,
    analytic_cost_to_go,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)

logger.disable("irsim")

COARSE = 0

# robot_state column layout (shared across the 14-robot system).
_PX, _PY, _GX, _GY = 0, 1, 9, 10


def goal_distances_from_state(robot_state: np.ndarray) -> np.ndarray:
    """Per-robot goal distance straight off an (N, 11) ``robot_state``."""
    s = np.asarray(robot_state, dtype=np.float64)
    return np.linalg.norm(s[:, [_GX, _GY]] - s[:, [_PX, _PY]], axis=1)


def backfill_returns(
    episodes: list[dict], keep_censored: bool = False
) -> dict[str, np.ndarray]:
    """
    Turn per-episode step records into flat, aligned label arrays.

    Each episode dict carries ``step_costs`` (realised cost of each decision,
    in order), ``h`` (the analytic heuristic at each decision *state*), the
    per-decision feature blocks, and ``reached`` (did the episode solve).

    The return at decision ``t`` is the suffix sum ``Σ_{s ≥ t} step_cost_s`` —
    the cost actually paid from that state onward, which is exactly what the
    leaf heuristic is trying to predict.  Pure and sim-free so the alignment
    and censoring rules are unit-testable.

    Returns flat arrays over all decisions, with ``valid`` marking the
    uncensored ones (``ratio`` is NaN where invalid).
    """
    out: dict[str, list] = {
        "episode": [], "t": [], "h": [], "G": [], "n_unreached": [],
        "sum_dist": [], "mode": [], "group": [], "valid": [],
        "group_feats": [], "global_feats": [],
    }
    for ep in episodes:
        costs = np.asarray(ep["step_costs"], dtype=np.float64)
        if costs.size == 0:
            continue
        # Suffix sum: G[t] = cost of decision t and everything after it.
        G = np.cumsum(costs[::-1])[::-1]
        valid = bool(ep["reached"])
        if not valid and not keep_censored:
            continue
        for t in range(costs.size):
            out["episode"].append(ep["index"])
            out["t"].append(t)
            out["h"].append(ep["h"][t])
            out["G"].append(float(G[t]))
            out["n_unreached"].append(ep["n_unreached"][t])
            out["sum_dist"].append(ep["sum_dist"][t])
            out["mode"].append(ep["mode"][t])
            out["group"].append(ep["group"][t])
            out["valid"].append(valid)
            out["group_feats"].append(ep["group_feats"][t])
            out["global_feats"].append(ep["global_feats"][t])

    arrays = {
        "episode": np.asarray(out["episode"], dtype=np.int32),
        "t": np.asarray(out["t"], dtype=np.int32),
        "h": np.asarray(out["h"], dtype=np.float64),
        "G": np.asarray(out["G"], dtype=np.float64),
        "n_unreached": np.asarray(out["n_unreached"], dtype=np.int32),
        "sum_dist": np.asarray(out["sum_dist"], dtype=np.float64),
        "mode": np.asarray(out["mode"], dtype=np.int8),
        "group": np.asarray(out["group"], dtype=np.int16),
        "valid": np.asarray(out["valid"], dtype=bool),
    }
    if out["group_feats"]:
        arrays["group_feats"] = np.stack(out["group_feats"]).astype(np.float32)
        arrays["global_feats"] = np.stack(out["global_feats"]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(arrays["h"] > 0.0, arrays["G"] / arrays["h"], np.nan)
    arrays["ratio"] = np.where(arrays["valid"], ratio, np.nan)
    return arrays


class LeafCollector:
    """
    Observer wired into ``eval_gaz14_lazy.run``'s ``on_step`` hook.

    Reads the decision *state* from ``env._robot_state`` before the step is
    applied, so ``h`` is the heuristic at the state the search planned from —
    the quantity the leaf evaluator is asked for.
    """

    def __init__(self, env, alpha: float, goal_threshold: float,
                 feature_builder: GroupFeatureBuilder) -> None:
        self.env = env
        self.alpha = float(alpha)
        self.goal_threshold = float(goal_threshold)
        self.feature_builder = feature_builder
        self.episodes: list[dict] = []
        self._current: dict | None = None
        self._pending_state: np.ndarray | None = None

    def before_decision(self) -> None:
        """Snapshot the pre-step state (the decision's own state)."""
        self._pending_state = np.asarray(self.env._robot_state, dtype=np.float64).copy()

    def _new_episode(self, index: int) -> dict:
        return {
            "index": index, "step_costs": [], "h": [], "n_unreached": [],
            "sum_dist": [], "mode": [], "group": [],
            "group_feats": [], "global_feats": [], "reached": False,
        }

    def __call__(self, ep: int, decision: dict, step_cost: float,
                 info: dict, done: bool) -> None:
        if self._current is None or self._current["index"] != ep:
            self._current = self._new_episode(ep)
            self.episodes.append(self._current)

        state = self._pending_state
        dist = goal_distances_from_state(state)
        unreached = dist > self.goal_threshold

        rec = self._current
        rec["step_costs"].append(float(step_cost))
        rec["h"].append(
            analytic_cost_to_go(dist, self.goal_threshold, self.alpha)
        )
        rec["n_unreached"].append(int(unreached.sum()))
        rec["sum_dist"].append(float(dist[unreached].sum()))
        rec["mode"].append(int(decision["mode"]))
        rec["group"].append(-1 if decision["group"] is None else int(decision["group"]))
        # Reuse the features the decision already computed when available.
        gf = decision.get("group_feats")
        glf = decision.get("global_feats")
        if gf is None or glf is None:
            gf = np.zeros((len(MOVE_GROUPS), 0), dtype=np.float32)
            glf = np.zeros((0,), dtype=np.float32)
        rec["group_feats"].append(np.asarray(gf, dtype=np.float32))
        rec["global_feats"].append(np.asarray(glf, dtype=np.float32))

        if done:
            rec["reached"] = bool(info.get("all_reached"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--variant", choices=("eager", "lazy"), default="eager",
                    help="which switcher generates the trajectories whose "
                         "realised cost becomes the label")
    ap.add_argument("--budget", type=int, default=115)
    ap.add_argument("--m", type=int, default=None,
                    help="Gumbel-top-m (default: 4 eager / 16 lazy)")
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, required=True,
                    help="directory for the .npz shard")
    ap.add_argument("--keep-censored", action="store_true",
                    help="also store timeout/collision episodes (valid=False); "
                         "excluded from the ratio either way")
    # --- policy that generates the trajectories -------------------------
    ap.add_argument("--prior-model", type=str, default=None,
                    help="learned PriorNet checkpoint; a better policy makes "
                         "the realised returns more representative of what "
                         "you will actually deploy")
    ap.add_argument("--feas-margin", type=float, default=0.0,
                    help="advisory in-tree feasibility threshold (lazy only; "
                         "the eager search vets for real and ignores it)")
    ap.add_argument("--value-model", type=str, default=None,
                    help="learned cost-to-go used as the search's LEAF. Note "
                         "this changes the policy being measured, not the "
                         "heuristic being scored — see the printed note")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt,
    )
    alpha = analytic_alpha(cost.precise_unit, coarse.lin_max, coarse.step_time)
    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)

    if args.variant == "eager":
        from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel_eager import (
            GumbelSwitcher14Eager as Switcher,
        )
        m = 4 if args.m is None else args.m
    else:
        from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import (
            GumbelSwitcher14 as Switcher,
        )
        m = 16 if args.m is None else args.m

    # ---- the policy whose realised cost becomes the label ---------------
    prior = None
    if args.prior_model:
        from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
            LearnedPrior,
            PriorNet,
        )

        prior = LearnedPrior(
            PriorNet.load(args.prior_model, map_location=device),
            feature_builder, feas_margin=args.feas_margin, device=device,
        )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import (
            LearnedCostToGo,
            load_value_checkpoint,
        )

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        _, vckpt = load_value_checkpoint(args.value_model, device=device)
        trained_n = (vckpt.get("meta", {}) or {}).get("num_robots")

    policy = Switcher(
        backbone=env.backbone, coarse=coarse, sim=sim, prior=prior,
        budget=args.budget, m=m, gumbel_scale=args.gumbel_scale,
        seed=args.seed, d_safe=args.d_safe,
        selection_interval=env.selection_interval,
        goal_threshold=args.goal_threshold, cost=env.cost,
        leaf_value=leaf_value, feature_builder=feature_builder,
    )

    search_leaf = "learned" if leaf_value is not None else "analytic"
    print(
        f"Env: {sim.num_robots} robots, {len(MOVE_GROUPS)} coarse groups, "
        f"variant={args.variant}, budget={args.budget}, m={m}, "
        f"episodes={args.episodes}, device={device}\n"
        f"Policy:  prior={args.prior_model or 'uniform'}  "
        f"search leaf={search_leaf}\n"
        f"Scored heuristic: analytic, alpha={alpha:.1f} cost per robot-metre "
        f"(precise pricing; precise_unit={cost.precise_unit})"
    )

    # The two roles are easy to conflate, so say it plainly.
    if leaf_value is not None:
        print(
            "\nNOTE: --value-model changes the POLICY, not the heuristic being\n"
            "  scored.  `h` in the shard is always the analytic heuristic, so\n"
            "  the ratio still answers 'how wrong is the analytic leaf', but\n"
            "  now against the returns of a policy that already plans with a\n"
            "  learned leaf.  For the cleanest read on whether to replace the\n"
            "  analytic leaf, collect WITHOUT --value-model as well and\n"
            "  compare: that is the configuration you would be improving from."
        )
        if trained_n is not None and int(trained_n) != sim.num_robots:
            print(
                f"  WARNING: that checkpoint was trained on {trained_n}-robot "
                f"episodes but this world has {sim.num_robots}.  Its congestion "
                f"model is out of distribution here, so the trajectories it "
                f"produces are not a trustworthy basis for labels."
            )

    collector = LeafCollector(env, alpha, args.goal_threshold, feature_builder)

    def decide(e):
        collector.before_decision()
        return policy.decide(e._robot_state)

    stats = run(env, decide, args.episodes, args.seed, policy=policy,
                on_step=collector)

    data = backfill_returns(collector.episodes, keep_censored=args.keep_censored)
    n_solved = sum(1 for e in collector.episodes if e["reached"])
    print(
        f"\n{stats['success_rate']:.1%} success  "
        f"{stats['collision_rate']:.1%} collision  "
        f"{stats['timeout_rate']:.1%} timeout\n"
        f"{n_solved}/{len(collector.episodes)} episodes solved -> "
        f"{int(data['valid'].sum())} labelled decisions"
    )
    if int(data["valid"].sum()) == 0:
        raise SystemExit(
            "No episode reached the goal, so no decision has a valid return. "
            "Raise --budget or --episodes, or check the run's success rate."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Provenance goes in the filename as well as the payload: labels are only
    # valid for the policy that produced them, and shards from different
    # policies must not silently merge into one fit.
    tag = f"{'p' if args.prior_model else 'u'}{search_leaf[0]}"
    path = out_dir / f"leaf_{args.variant}_b{args.budget}_{tag}_s{args.seed}.npz"
    np.savez_compressed(
        path, alpha=np.float64(alpha),
        goal_threshold=np.float64(args.goal_threshold),
        precise_unit=np.float64(cost.precise_unit),
        variant=str(args.variant), budget=np.int32(args.budget),
        policy=str(f"{args.variant}/b{args.budget}/m{m}/"
                   f"prior={Path(args.prior_model).name if args.prior_model else 'uniform'}/"
                   f"leaf={search_leaf}"),
        **data,
    )
    print(f"Wrote {path}")

    valid = data["valid"]
    r = data["ratio"][valid]
    print(
        f"\nQuick look — G / h_analytic over {r.size} labelled decisions:\n"
        f"  median {np.median(r):.4f}   mean {r.mean():.4f}   "
        f"IQR [{np.percentile(r, 25):.4f}, {np.percentile(r, 75):.4f}]\n"
        f"  => the heuristic overestimates by about "
        f"{1.0 / max(np.median(r), 1e-9):.1f}x at the median\n"
        f"Run analyze_leaf_data.py for the constant-alpha test."
    )


if __name__ == "__main__":
    main()
