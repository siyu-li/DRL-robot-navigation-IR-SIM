"""
Feasibility diagnostic for the two CAPSwitcher-14 action sets.

``eval_mpc_14``'s ``safe-coarse avail.`` row is an **OR over the whole coarse
action set**: the fraction of decisions at which *at least one* of the 22 groups
passed the shield.  At ~98% it says the switcher is essentially never left
without a coarse option — it says nothing about how feasible an individual group
is, and nothing at all about precise, which is never shield-vetted.

This harness reports the missing rates, scoring both action sets with the *same*
criterion the shield applies to coarse (swept min clearance >= ``d_safe``, see
``rl/shield.py``):

* **coarse** — vet all 22 groups at every decision: per-group feasibility rate,
  the distribution of how many groups are simultaneously safe, and the any-safe
  rate (which reproduces ``safe-coarse avail.`` as a cross-check).
* **precise** — roll the decision out with ``ForwardModel14.precise_rollout``
  and apply the shield criterion to the driven robot at every sub-step: the
  feasibility rate precise *would* have had, had anything vetted it.

Two asymmetries in the current design are what the comparison exposes:

* coarse is vetted over its **whole swept path** against a ``d_safe`` margin;
  precise is only tested at its **end state** against a **zero** margin
  (``collision_pred`` in ``rl/search/tree.py::make_node``).  The
  ``endpoint clean, swept breach`` row measures that blind spot directly.
* one coarse decision moves ``|members|`` robots by ``move_distance``; one
  precise decision moves *every* unreached robot for ``selection_interval``
  sub-steps.  Precise therefore buys far more exposure per decision, so rates
  are also reported per robot-metre of motion.

Realized clearance from the live sim is tracked alongside the model-side vet, so
the model numbers can be checked against what the simulator actually did.

Usage (GPU box — local irsim ``step`` crashes; see project memory):
    python -m robot_nav.eval_feasibility_14 --episodes 20 --budget 100
    python -m robot_nav.eval_feasibility_14 --episodes 20 --budget 100 \
        --prior-model <ckpt> --value-model <ckpt>
    python -m robot_nav.eval_feasibility_14 --episodes 20 --drive coarse
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger

from robot_nav.eval_mpc_14 import (
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_COST_CONFIG,
    build_env,
    resolve_device,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.shield import (
    ShieldGeometry,
    min_member_clearance,
)
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    ForwardModel14,
    build_forward_model,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import GumbelSwitcher14

logger.disable("irsim")

COARSE, PRECISE = 0, 1


# ---------------------------------------------------------------------------
# Per-decision vetting
# ---------------------------------------------------------------------------


@dataclass
class DecisionVet:
    """Both action sets scored at one decision state."""

    coarse_clearance: np.ndarray   # (22,) swept min clearance per group (m)
    coarse_motion: np.ndarray      # (22,) robot-metres the group's move commits
    precise_driven: float          # swept min clearance of the driven robot (m)
    precise_allpairs: float        # swept min clearance over all pairs (m)
    precise_endpoint: float        # end-state min clearance — the planner's view
    precise_motion: float          # robot-metres the precise rollout commits
    precise_substeps: int          # sub-steps the rollout executes


def vet_precise(model: ForwardModel14, ms) -> tuple[float, float, float, float, int]:
    """
    Score one precise-all decision with the shield's own clearance criterion.

    Precise drives exactly one robot per sub-step, so the shield's "only score
    pairs the move puts at risk" rule maps onto it directly: at sub-step *k* the
    driven robot is the sole member, and its segment spans the positions before
    and after the sub-step — the same sampling ``swept_positions`` hands the
    shield for coarse.

    Returns:
        ``(driven, allpairs, endpoint, motion, substeps)`` — the driven-robot
        swept minimum (the coarse-comparable number), the all-pairs swept
        minimum over post-motion states (what ``collision_pred`` would have
        found had it run at every sub-step rather than only at the end), the
        end-state all-pairs minimum (what it actually checks), the robot-metres
        of motion committed, and the sub-step count.
    """
    _, path = model.precise_rollout(ms)
    all_robots = np.arange(model.N)

    driven = allpairs = float("inf")
    motion = 0.0
    for k in range(1, len(path)):
        r, pos = path[k]
        prev = path[k - 1][1]
        segment = np.stack([prev, pos], axis=0)                 # (2, N, 2)
        driven = min(driven, min_member_clearance(segment, [r], model.geom))
        allpairs = min(
            allpairs, min_member_clearance(pos[None, :, :], all_robots, model.geom)
        )
        motion += float(np.linalg.norm(pos[r] - prev[r]))

    endpoint = (
        min_member_clearance(path[-1][1][None, :, :], all_robots, model.geom)
        if path
        else float("inf")
    )
    return driven, allpairs, endpoint, motion, len(path) - 1


def vet_decision(model: ForwardModel14, ms, coarse) -> DecisionVet:
    """Vet every coarse group and the precise rollout at one decision state."""
    groups = coarse.selectable_groups()
    clearance = np.full(len(MOVE_GROUPS), np.nan, dtype=np.float64)
    motion = np.zeros(len(MOVE_GROUPS), dtype=np.float64)
    for g in groups:
        clearance[g] = model.coarse_move(ms, g).candidate.clearance
        motion[g] = coarse.members_of(g).size * coarse.move_distances[g]

    driven, allpairs, endpoint, p_motion, p_substeps = vet_precise(model, ms)
    return DecisionVet(
        coarse_clearance=clearance,
        coarse_motion=motion,
        precise_driven=driven,
        precise_allpairs=allpairs,
        precise_endpoint=endpoint,
        precise_motion=p_motion,
        precise_substeps=p_substeps,
    )


# ---------------------------------------------------------------------------
# Realized (sim-side) clearance
# ---------------------------------------------------------------------------


class RealizedClearance:
    """
    Record the clearance the *simulator* actually reached, per executed sub-step.

    Wraps ``SwitcherEnv._apply_substep`` on the instance (the env has no hook)
    and, after each sub-step, measures the all-pairs minimum clearance of the
    live poses with the shield's geometry.  This is ground truth: it validates —
    or refutes — the analytic model's swept numbers.
    """

    def __init__(self, env) -> None:
        self.env = env
        self._orig = env._apply_substep
        self.geom: ShieldGeometry | None = None
        self.mode: int | None = None
        self.worst: float = float("inf")   # current decision's minimum
        self.motion: float = 0.0           # current decision's robot-metres
        self._prev_xy: np.ndarray | None = None

        def wrapped(sim_actions, info):
            done = self._orig(sim_actions, info)
            self._observe()
            return done

        env._apply_substep = wrapped

    def restore(self) -> None:
        self.env._apply_substep = self._orig

    def begin_decision(self, geom: ShieldGeometry, mode: int) -> None:
        self.geom = geom
        self.mode = mode
        self.worst = float("inf")
        self.motion = 0.0
        self._prev_xy = self._xy()

    def _xy(self) -> np.ndarray:
        return np.asarray(
            [[float(p[0]), float(p[1])] for p in self.env._poses], dtype=np.float64
        )

    def _observe(self) -> None:
        if self.geom is None:
            return
        xy = self._xy()
        self.worst = min(
            self.worst,
            min_member_clearance(xy[None, :, :], np.arange(xy.shape[0]), self.geom),
        )
        if self._prev_xy is not None:
            self.motion += float(
                np.linalg.norm(xy - self._prev_xy, axis=1).sum()
            )
        self._prev_xy = xy


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


@dataclass
class Accum:
    """Everything the run collects, flattened at report time."""

    coarse_clearance: list = field(default_factory=list)   # per decision, (22,)
    precise_driven: list = field(default_factory=list)
    precise_allpairs: list = field(default_factory=list)
    precise_endpoint: list = field(default_factory=list)
    precise_motion: list = field(default_factory=list)
    precise_substeps: list = field(default_factory=list)
    coarse_motion: list = field(default_factory=list)      # per decision, (22,)
    chosen_mode: list = field(default_factory=list)
    chosen_group: list = field(default_factory=list)
    realized_worst: list = field(default_factory=list)     # (mode, clearance)
    realized_motion: list = field(default_factory=list)    # (mode, robot-metres)
    collided: list = field(default_factory=list)           # (mode, bool)
    episodes: int = 0
    n_success: int = 0
    n_collision: int = 0
    n_timeout: int = 0


def run(
    env, coarse, sim, decide_fn, episodes: int, base_seed: int,
    d_safe: float, selection_interval: int, goal_threshold: float,
    cost: SwitcherCost, vet_every: int,
) -> Accum:
    """
    Drive ``episodes`` seeded episodes with ``decide_fn``, vetting both action
    sets at every ``vet_every``-th decision *before* the chosen action executes.

    The vet uses its own :class:`ForwardModel14` built from the live sim, so it
    is independent of what the driving policy happened to materialise — and it
    is charged to nothing: ``precise_rollout`` is deliberately outside the
    transition budget.
    """
    acc = Accum(episodes=episodes)
    tracker = RealizedClearance(env)
    try:
        for ep in range(episodes):
            seed = base_seed + ep
            seed_episode(env, seed)

            env.reset()
            done = False
            k = 0
            info: dict = {}
            while not done:
                decision = decide_fn(env)
                mode = decision["mode"]

                if k % vet_every == 0:
                    model = build_forward_model(
                        env.backbone, coarse, sim, env._robot_state,
                        d_safe=d_safe,
                        selection_interval=selection_interval,
                        goal_threshold=goal_threshold,
                        cost=cost,
                        default_rho=0.2,
                    )
                    ms = ForwardModel14.state_from_robot_state(env._robot_state)
                    v = vet_decision(model, ms, coarse)
                    acc.coarse_clearance.append(v.coarse_clearance)
                    acc.coarse_motion.append(v.coarse_motion)
                    acc.precise_driven.append(v.precise_driven)
                    acc.precise_allpairs.append(v.precise_allpairs)
                    acc.precise_endpoint.append(v.precise_endpoint)
                    acc.precise_motion.append(v.precise_motion)
                    acc.precise_substeps.append(v.precise_substeps)
                    acc.chosen_mode.append(mode)
                    acc.chosen_group.append(
                        -1 if decision["group"] is None else int(decision["group"])
                    )

                tracker.begin_decision(ShieldGeometry.from_sim(sim), mode)
                _, _, done, info = env.step(
                    mode, group=decision["group"], frames=decision["frames"]
                )
                acc.realized_worst.append((mode, tracker.worst))
                acc.realized_motion.append((mode, tracker.motion))
                acc.collided.append((mode, bool(info["collision"])))
                k += 1

            acc.n_success += bool(info.get("all_reached"))
            acc.n_collision += bool(info.get("collision"))
            acc.n_timeout += bool(info.get("timeout"))
            print(
                f"  ep {ep + 1}/{episodes}: {k} decisions, "
                f"{'success' if info.get('all_reached') else ''}"
                f"{'collision' if info.get('collision') else ''}"
                f"{'timeout' if info.get('timeout') else ''}",
                flush=True,
            )
    finally:
        tracker.restore()
    return acc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _q(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else float("nan")


def report(acc: Accum, coarse, d_safe: float) -> None:
    """Print the feasibility tables."""
    if not acc.coarse_clearance:
        print("\nNo decisions were vetted — nothing to report.\n")
        return
    C = np.stack(acc.coarse_clearance)
    Cm = np.stack(acc.coarse_motion)
    groups = coarse.selectable_groups()
    C = C[:, groups]
    Cm = Cm[:, groups]
    n_dec = C.shape[0]

    safe = C >= d_safe                      # (decisions, groups)
    n_safe = safe.sum(axis=1)

    pd_ = np.asarray(acc.precise_driven)
    pa = np.asarray(acc.precise_allpairs)
    pe = np.asarray(acc.precise_endpoint)
    pm = np.asarray(acc.precise_motion)
    ps = np.asarray(acc.precise_substeps)

    print("\n" + "=" * 78)
    print(f"Action-set feasibility  —  {n_dec} vetted decisions, "
          f"{acc.episodes} episodes, d_safe = {d_safe} m")
    print(f"outcome: {acc.n_success}/{acc.episodes} success, "
          f"{acc.n_collision} collision, {acc.n_timeout} timeout")
    print("=" * 78)

    print("\nCOARSE action set  (22 groups, shield criterion: swept min clearance "
          f">= {d_safe})")
    print(f"  any group safe (= 'safe-coarse avail.')   {safe.any(axis=1).mean():>8.1%}")
    print(f"  per-group feasibility (pooled)            "
          f"{safe.mean():>8.1%}   <- the rate the OR row hides")
    print(f"  safe groups per decision                  "
          f"{n_safe.mean():>8.2f} / {len(groups)}")
    print(f"    median / p10 / min                      "
          f"{_q(n_safe, 50):>8.0f} {_q(n_safe, 10):>6.0f} {n_safe.min():>6.0f}")
    print(f"  decisions with 0 safe groups              "
          f"{(n_safe == 0).mean():>8.1%}")
    print(f"  decisions with <= 2 safe groups           "
          f"{(n_safe <= 2).mean():>8.1%}")
    print(f"  clearance p50 / p10 / min (m)             "
          f"{_q(C, 50):>8.2f} {_q(C, 10):>6.2f} {C.min():>6.2f}")
    print(f"  robot-metres per coarse decision          {Cm.mean():>8.2f}")

    print("\nPRECISE action set  (never vetted in the current design; scored here "
          "with the same criterion)")
    print(f"  feasible @ d_safe={d_safe} (driven robot)   "
          f"{(pd_ >= d_safe).mean():>8.1%}   <- coarse-comparable")
    print(f"  feasible @ 0 margin (driven robot)        {(pd_ >= 0).mean():>8.1%}")
    print(f"  no contact at any sub-step (all pairs)    {(pa >= 0).mean():>8.1%}")
    print(f"  no contact at END state (all pairs)       "
          f"{(pe >= 0).mean():>8.1%}   <- all the planner checks")
    print(f"  BLIND SPOT: end clean but sub-step breach "
          f"{((pe >= 0) & (pa < 0)).mean():>8.1%}")
    print(f"  driven clearance p50 / p10 / min (m)      "
          f"{_q(pd_, 50):>8.2f} {_q(pd_, 10):>6.2f} {pd_.min():>6.2f}")
    print(f"  all-pairs clearance p50 / p10 / min (m)   "
          f"{_q(pa, 50):>8.2f} {_q(pa, 10):>6.2f} {pa.min():>6.2f}")
    print(f"  sub-steps per precise decision            {ps.mean():>8.1f}")
    print(f"  robot-metres per precise decision         {pm.mean():>8.2f}")

    # Exposure-normalised: coarse commits a fraction of the motion precise does,
    # so raw per-decision rates flatter coarse. Compare infeasibility per metre.
    coarse_inf_per_m = (~safe).mean() / max(Cm.mean(), 1e-9)
    precise_inf_per_m = float((pd_ < d_safe).mean()) / max(pm.mean(), 1e-9)
    print("\nEXPOSURE-NORMALISED  (infeasible fraction per robot-metre committed)")
    print(f"  coarse    {coarse_inf_per_m:>8.4f} /m")
    print(f"  precise   {precise_inf_per_m:>8.4f} /m")

    # ---- per-group breakdown ------------------------------------------
    print(f"\nPER-GROUP feasibility ({len(groups)} groups)")
    print(f"  {'grp':>4} {'size':>5} {'feasible':>10} {'clr p50':>9} "
          f"{'clr p10':>9} {'move m':>8}")
    order = np.argsort(-safe.mean(axis=0))
    for j in order:
        g = groups[j]
        print(f"  {g:>4} {coarse.members_of(g).size:>5} "
              f"{safe[:, j].mean():>9.1%} {_q(C[:, j], 50):>9.2f} "
              f"{_q(C[:, j], 10):>9.2f} {coarse.move_distances[g]:>8.2f}")

    # ---- what the switcher actually chose ------------------------------
    chosen = np.asarray(acc.chosen_mode)
    if chosen.size:
        cm = chosen == COARSE
        pmask = chosen == PRECISE
        print("\nAT THE DECISIONS THE SWITCHER ACTUALLY MADE")
        print(f"  chose coarse   {cm.mean():>8.1%}  "
              f"(precise was feasible there {((pd_ >= d_safe) & cm).sum()}/{cm.sum()})")
        print(f"  chose precise  {pmask.mean():>8.1%}  "
              f"(precise was feasible there "
              f"{((pd_ >= d_safe) & pmask).sum()}/{pmask.sum()})")
        if pmask.any():
            print(f"  precise chosen while 0 coarse groups safe   "
                  f"{((n_safe == 0) & pmask).sum()}")
            print(f"  precise chosen while some coarse group safe "
                  f"{((n_safe > 0) & pmask).sum()}")

    # ---- realized (sim ground truth) -----------------------------------
    rw = np.asarray([w for _, w in acc.realized_worst], dtype=np.float64)
    rmode = np.asarray([m for m, _ in acc.realized_worst])
    rmot = np.asarray([m for _, m in acc.realized_motion], dtype=np.float64)
    coll = np.asarray([c for _, c in acc.collided])
    print("\nREALIZED IN THE SIMULATOR (all executed decisions, ground truth)")
    print(f"  {'mode':>8} {'decisions':>10} {'collisions':>11} {'clr p50':>9} "
          f"{'clr p10':>9} {'clr min':>9} {'robot-m/dec':>12} {'coll/100 robot-m':>18}")
    for name, mask in (("coarse", rmode == COARSE), ("precise", rmode == PRECISE)):
        if not mask.any():
            continue
        tot_m = rmot[mask].sum()
        print(f"  {name:>8} {mask.sum():>10} {coll[mask].sum():>11} "
              f"{_q(rw[mask], 50):>9.2f} {_q(rw[mask], 10):>9.2f} "
              f"{rw[mask].min():>9.2f} {rmot[mask].mean():>12.2f} "
              f"{100.0 * coll[mask].sum() / max(tot_m, 1e-9):>18.3f}")
    print()


# ---------------------------------------------------------------------------
# Deciders
# ---------------------------------------------------------------------------


def _precise_only(env) -> dict:
    return {"mode": PRECISE, "group": None, "frames": None}


def _coarse_only(env) -> dict:
    return {"mode": COARSE, "group": None, "frames": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--drive", choices=["gumbel", "coarse", "precise"],
                    default="gumbel",
                    help="policy that drives the episodes (sets the state "
                         "distribution the action sets are vetted on)")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--vet-every", type=int, default=1,
                    help="vet every k-th decision (the precise rollout is a "
                         "full GAT rollout — raise this to trade resolution "
                         "for wall-clock)")
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--prior-model", type=str, default=None)
    ap.add_argument("--feas-margin", type=float, default=0.0)
    ap.add_argument("--value-model", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt,
    )
    print(f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
          f"{len(MOVE_GROUPS)} coarse groups, d_safe={args.d_safe}, "
          f"drive={args.drive}, device={device}")

    if args.drive == "coarse":
        decide_fn = _coarse_only
    elif args.drive == "precise":
        decide_fn = _precise_only
    else:
        leaf_value = None
        if args.value_model:
            from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

            leaf_value = LearnedCostToGo(args.value_model, device=device)
        feature_builder = GroupFeatureBuilder(MOVE_GROUPS)
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
        policy = GumbelSwitcher14(
            backbone=env.backbone, coarse=coarse, sim=sim, prior=prior,
            budget=args.budget, m=args.m, c_visit=args.c_visit,
            c_scale=args.c_scale, gumbel_scale=args.gumbel_scale, seed=args.seed,
            d_safe=args.d_safe, selection_interval=env.selection_interval,
            goal_threshold=args.goal_threshold, cost=env.cost,
            leaf_value=leaf_value, feature_builder=feature_builder,
        )
        decide_fn = lambda e: policy.decide(e._robot_state)  # noqa: E731

    print(f"\nRunning {args.episodes} episodes ...")
    acc = run(
        env, coarse, sim, decide_fn, args.episodes, args.seed,
        d_safe=args.d_safe, selection_interval=env.selection_interval,
        goal_threshold=args.goal_threshold, cost=cost, vet_every=args.vet_every,
    )
    report(acc, coarse, args.d_safe)


if __name__ == "__main__":
    main()
