"""
Evaluation + phase-0 collection harness for the 14-robot lazy Gumbel switcher.

Runs ``GumbelSwitcher14`` (22 coarse groups + precise, lazy tree, budget in
model transitions) against the two mode-only baselines over matched seeded
episodes.  With ``--log-pi-targets`` every real decision is logged as
training data for the learned prior:

    features (root group/global), all 22 shield clearances (exact labels for
    the feasibility head), the legal mask, and the improved root policy π′
    (root-only distillation target, settled).

Phase 0 = ``--episodes N --budgets 100 --log-pi-targets data/pi_targets_14``
with the default uniform prior (high-budget teacher).  Iteration t trains a
prior with ``train_prior.py`` and re-collects with ``--prior-model <ckpt>``.

Usage (run on the GPU box — local irsim step crashes; see project memory):
    python -m robot_nav.eval_mpc_14 --episodes 100 --budgets 100  \
        --log-pi-targets data/pi_targets_14
    python -m robot_nav.eval_mpc_14 --episodes 100 --budgets 40 100 \
        --prior-model <ckpt> --value-model <ckpt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.switcher_env import (
    SwitcherEnv,
    seed_episode,
)
from robot_nav.models.MARL.capswitcher_14.configs import (
    MOVE_GROUPS,
    make_coarse_steering,
)
from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
    CoarseSteering14,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import GumbelSwitcher14

logger.disable("irsim")

COARSE, PRECISE = 0, 1

# Reported metrics, shared by this harness and the plan→distil loop.
#
# The headline is **cost per successful episode** — `avg cost/ep` mixes the
# solved and unsolved episodes together, so it moves with the success/timeout
# composition as much as with plan quality, and it is not comparable across
# cost tables at all.  The cost split below separates the flat per-group coarse
# constants from the `precise_unit` surcharge, which is the term that changes
# between cost configs.
RESULT_ROWS = [
    ("success rate",         "success_rate",        "{:.1%}"),
    ("collision rate",       "collision_rate",      "{:.1%}"),
    ("timeout rate",         "timeout_rate",        "{:.1%}"),
    ("avg decisions/ep",     "avg_decisions",       "{:.1f}"),
    ("cost/success ep  <<",  "avg_cost_success",    "{:.0f}"),
    ("  coarse cost/ep",     "avg_coarse_cost",     "{:.0f}"),
    ("  precise cost/ep",    "avg_precise_cost",    "{:.0f}"),
    ("  precise share",      "precise_cost_share",  "{:.1%}"),
    ("avg cost/ep (mixed)",  "avg_cost",            "{:.0f}"),
    ("precise fraction",     "precise_frac",        "{:.1%}"),
    ("coarse fraction",      "coarse_frac",         "{:.1%}"),
    ("safe-coarse avail.",   "safe_avail_frac",     "{:.1%}"),
    ("coarse breaches (!)",  "coarse_breach",       "{:d}"),
    ("precise breaches",     "precise_breach",      "{:d}"),
    ("  driven robot",       "precise_breach_active",    "{:d}"),
    ("  bystander (!)",      "precise_breach_bystander", "{:d}"),
    ("avg transitions/dec",  "avg_transitions",     "{:.1f}"),
]

DEFAULT_BACKBONE_CKPT = (
    "robot_nav/models/MARL/marlTD3/checkpoint/"
    "Mar.04_obstacle_14robots_partial_inactive/"
    "TD3-MARL-obstacle-14robots-partial-inactive_epoch210"
)


def resolve_device(name: str) -> torch.device:
    """Resolve ``--device``; ``"auto"`` picks cuda when available, else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


DEFAULT_COST_CONFIG = "robot_nav/models/MARL/capswitcher_14/cost_14robots.yaml"


def build_env(
    device: torch.device, cost: SwitcherCost, goal_threshold: float,
    backbone_ckpt: str, disable_plotting: bool = True,
) -> tuple[SwitcherEnv, CoarseSteering14, MARL_SIM_OBSTACLE]:
    """
    Construct 14-robot sim + backbone + coarse primitive + switcher env.

    Plotting is off by default so every evaluation path stays headless;
    ``render_gaz14.py`` turns it on to watch a policy live.
    """
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
        disable_plotting=disable_plotting,
        reward_phase=6,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=1.5,
        num_inactive_robots=0,
    )
    assert sim.num_robots == 14, f"expected 14 robots, world has {sim.num_robots}"
    backbone = GATBackbone(
        checkpoint_path=Path(backbone_ckpt),
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        device=device,
        embedding_source="decoder",
    )
    coarse = make_coarse_steering(
        move_distance=cost.move_distances,
        method="nonlinear",
        step_time=sim.env.step_time,
        ang_max=1.0,
        lin_max=0.5,
    )
    # Fail loudly if the cost YAML's group table drifted from the algebra.
    cost.validate_members(
        {g: [int(i) for i in coarse.members_of(g)]
         for g in coarse.selectable_groups()}
    )
    env = SwitcherEnv(
        sim=sim,
        backbone=backbone,
        coarse_steering=coarse,
        selection_interval=5,
        max_decisions=120,
        cost=cost,
        goal_threshold=goal_threshold,
        device=device,
        terminate_on_oob=False,
    )
    return env, coarse, sim


def run(
    env: SwitcherEnv, decide_fn, episodes: int, base_seed: int, policy=None
) -> dict:
    """
    Run ``decide_fn`` for ``episodes`` seeded episodes and collect stats.

    ``policy`` (a ``GumbelSwitcher14``) supplies per-decision transition
    counts — the lazy budget unit; baselines report 0.

    Episode cost is split into its two priced components so a run is readable
    independently of how precision happens to be priced:

    * ``coarse_cost``   — the flat per-group constants (the baseline of moving
      the formation at all);
    * ``precise_cost``  — the ``precise_unit`` surcharge for resolving robots
      one at a time.

    Collisions are attributed to the mode that was executing when the sim
    flagged them (``coarse_breach`` / ``precise_breach``, which sum to the
    collision count), and — for precise — to whether the flagged robot was the
    one being driven or a stationary bystander.
    """
    if policy is not None:
        policy.decision_transitions = []
    n = {"success": 0, "collision": 0, "timeout": 0}
    coarse_dec = precise_dec = total_dec = 0
    safe_available = coarse_breach = precise_breach = 0
    precise_breach_active = precise_breach_bystander = 0
    costs, lengths, success_costs = [], [], []
    coarse_costs, precise_costs = [], []

    for ep in range(episodes):
        seed = base_seed + ep
        seed_episode(env, seed)

        env.reset()
        done = False
        ep_cost, ep_len = 0.0, 0
        ep_coarse_cost = ep_precise_cost = 0.0
        info: dict = {}
        while not done:
            decision = decide_fn(env)
            cands = decision.get("candidates") or []
            if any(getattr(c, "safe", False) for c in cands):
                safe_available += 1

            _, _, done, info = env.step(
                decision["mode"], group=decision["group"], frames=decision["frames"]
            )
            ep_len += 1
            total_dec += 1
            step_cost = float(info["path_cost"])
            ep_cost += step_cost
            if decision["mode"] == COARSE:
                coarse_dec += 1
                ep_coarse_cost += step_cost
                if info["collision"]:
                    coarse_breach += 1
            else:
                precise_dec += 1
                ep_precise_cost += step_cost
                if info["collision"]:
                    precise_breach += 1
                    # Precise drives exactly one robot at a time; the rest hold
                    # still.  If the flagged robot is the driven one, the frozen
                    # individual policy steered into something.
                    flagged = set(info.get("collision_robots") or [])
                    if info.get("active_robot") in flagged:
                        precise_breach_active += 1
                    else:
                        precise_breach_bystander += 1

        if info.get("all_reached"):
            n["success"] += 1
            success_costs.append(ep_cost)
        if info.get("collision"):
            n["collision"] += 1
        if info.get("timeout"):
            n["timeout"] += 1
        costs.append(ep_cost)
        coarse_costs.append(ep_coarse_cost)
        precise_costs.append(ep_precise_cost)
        lengths.append(ep_len)

    avg_cost = float(np.mean(costs))
    avg_coarse_cost = float(np.mean(coarse_costs))
    avg_precise_cost = float(np.mean(precise_costs))
    return {
        "episodes": episodes,
        "success_rate": n["success"] / episodes,
        "collision_rate": n["collision"] / episodes,
        "timeout_rate": n["timeout"] / episodes,
        "avg_decisions": float(np.mean(lengths)),
        "avg_cost": avg_cost,
        # Headline: what a solved episode actually costs.  None when no episode
        # succeeded (cycle 1 of a cold run can legitimately have zero).
        "avg_cost_success": (
            float(np.mean(success_costs)) if success_costs else None
        ),
        "avg_coarse_cost": avg_coarse_cost,
        "avg_precise_cost": avg_precise_cost,
        "precise_cost_share": avg_precise_cost / max(avg_cost, 1e-9),
        "coarse_frac": coarse_dec / max(total_dec, 1),
        "precise_frac": precise_dec / max(total_dec, 1),
        "safe_avail_frac": safe_available / max(total_dec, 1),
        "coarse_breach": coarse_breach,
        "precise_breach": precise_breach,
        "precise_breach_active": precise_breach_active,
        "precise_breach_bystander": precise_breach_bystander,
        "avg_transitions": (
            float(np.mean(policy.decision_transitions))
            if policy is not None and policy.decision_transitions
            else 0.0
        ),
    }


# ---- decision sources -----------------------------------------------------

def _precise_only(env: SwitcherEnv) -> dict:
    return {"mode": PRECISE, "group": None, "frames": None, "candidates": []}


def _coarse_only(env: SwitcherEnv) -> dict:
    # group=None → SwitcherEnv picks a uniform-random selectable group.
    return {"mode": COARSE, "group": None, "frames": None, "candidates": []}


def _gumbel_decider(policy: GumbelSwitcher14, pi_log: list | None):
    """Gumbel decider, optionally logging phase-0 prior-training samples."""

    def decide(env: SwitcherEnv) -> dict:
        decision = policy.decide(env._robot_state)
        if pi_log is not None:
            cands = decision.get("candidates") or []
            clearance = np.full(len(MOVE_GROUPS), np.nan, dtype=np.float32)
            safe = np.zeros(len(MOVE_GROUPS), dtype=bool)
            for c in cands:
                clearance[c.group] = c.clearance
                safe[c.group] = c.safe
            pi_log.append(
                {
                    "robot_state": np.asarray(env._robot_state, dtype=np.float32),
                    "group_feats": decision["group_feats"],
                    "global_feats": decision["global_feats"],
                    "pi_prime": np.asarray(decision["pi_prime"], dtype=np.float32),
                    "prior_logits": np.asarray(
                        decision["prior_logits"], dtype=np.float32
                    ),
                    "legal": np.asarray(decision["legal"], dtype=bool),
                    "clearance": clearance,
                    "safe": safe,
                    "n_transitions": decision["n_transitions"],
                }
            )
        return decision

    return decide


def _save_pi_targets(
    pi_log: list, out_dir: Path, name: str, d_safe: float
) -> None:
    """Save one fixed-shape .npz shard per eval row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        robot_states=np.stack([d["robot_state"] for d in pi_log]),
        group_feats=np.stack([d["group_feats"] for d in pi_log]),
        global_feats=np.stack([d["global_feats"] for d in pi_log]),
        pi_prime=np.stack([d["pi_prime"] for d in pi_log]),
        prior_logits=np.stack([d["prior_logits"] for d in pi_log]),
        legal=np.stack([d["legal"] for d in pi_log]),
        clearance=np.stack([d["clearance"] for d in pi_log]),
        safe=np.stack([d["safe"] for d in pi_log]),
        n_transitions=np.array([d["n_transitions"] for d in pi_log]),
        d_safe=np.float32(d_safe),
    )
    print(f"Saved {len(pi_log)} prior-training samples to {path}")


def print_table(results: dict[str, dict]) -> None:
    """Side-by-side comparison of the result dicts."""
    names = list(results)
    header = f"{'metric':<24}" + "".join(f"{nm:>14}" for nm in names)
    print("\n" + header)
    print("-" * len(header))
    for label, key, fmt in RESULT_ROWS:
        line = f"{label:<24}"
        for nm in names:
            v = results[nm].get(key)
            line += f"{'—' if v is None else fmt.format(v):>14}"
        print(line)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100],
                    help="transition budgets per decision (root's eager 23 count)")
    ap.add_argument("--m", type=int, default=16,
                    help="Gumbel-top-m root actions sampled without replacement")
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG,
                    help="SwitcherCost YAML (per-group move_distance + cost, "
                         "precise_unit)")
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto",
                    help="torch device for the backbone / prior / value nets "
                         "('auto' = cuda when available, else cpu)")
    ap.add_argument("--baselines", action="store_true",
                    help="also run precise-only and coarse-only baselines")
    ap.add_argument("--prior-model", type=str, default=None,
                    help="learned PriorNet checkpoint (train_prior.py); "
                         "default is the uniform phase-0 prior")
    ap.add_argument("--feas-margin", type=float, default=0.0,
                    help="advisory in-tree feasibility threshold (m) on the "
                         "predicted clearance margin")
    ap.add_argument("--value-model", type=str, default=None,
                    help="learned cost-to-go checkpoint (train_value.py)")
    ap.add_argument("--log-pi-targets", type=str, default=None,
                    help="directory for harvests the training data for learning prior")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt,
    )
    print(
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"{len(MOVE_GROUPS)} coarse groups, cost_config={args.cost_config}, "
        f"precise_unit={cost.precise_unit}, d_safe={args.d_safe}, "
        f"budgets={args.budgets}, prior={args.prior_model or 'uniform'}, "
        f"device={device}"
    )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        print(f"Learned leaf: feature={leaf_value.feature}, "
              f"precise_cost={leaf_value.precise_cost}")

    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)

    prior = None
    if args.prior_model:
        from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
            LearnedPrior,
            PriorNet,
        )

        prior = LearnedPrior(
            PriorNet.load(args.prior_model, map_location=device),
            feature_builder,
            feas_margin=args.feas_margin,
            device=device,
        )
        print(f"Learned prior: {args.prior_model} (feas_margin={args.feas_margin})")

    results: dict[str, dict] = {}

    if args.baselines:
        print(f"\nRunning precise-only for {args.episodes} episodes ...")
        results["precise"] = run(env, _precise_only, args.episodes, args.seed)
        print(f"Running coarse-only for {args.episodes} episodes ...")
        results["coarse"] = run(env, _coarse_only, args.episodes, args.seed)

    for b in args.budgets:
        policy = GumbelSwitcher14(
            backbone=env.backbone,
            coarse=coarse,
            sim=sim,
            prior=prior,
            budget=b,
            m=args.m,
            c_visit=args.c_visit,
            c_scale=args.c_scale,
            gumbel_scale=args.gumbel_scale,
            seed=args.seed,
            d_safe=args.d_safe,
            selection_interval=env.selection_interval,
            goal_threshold=args.goal_threshold,
            cost=env.cost,
            leaf_value=leaf_value,
            feature_builder=feature_builder,
        )
        tag = "+p" if args.prior_model else ""
        name = f"GAZ14-b{b}{tag}"
        pi_log = [] if args.log_pi_targets else None
        print(f"\nRunning {name} for {args.episodes} episodes ...")
        results[name] = run(env, _gumbel_decider(policy, pi_log),
                            args.episodes, args.seed, policy=policy)
        if pi_log:
            _save_pi_targets(pi_log, Path(args.log_pi_targets), name, args.d_safe)

    print_table(results)
    if any(r["coarse_breach"] > 0 for r in results.values()):
        print("WARNING: coarse breaches > 0 — raise --d-safe or check geometry.\n")


if __name__ == "__main__":
    main()
