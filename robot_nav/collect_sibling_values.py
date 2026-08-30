"""
Sibling resolution: exact off-path values from independent sub-searches
(``value_data_collection_plan.md`` §5).

For a stratified subsample of expanded nodes from A* trace shards, run an
independent A* from **each stored child state** to convergence and record the
per-child plan cost — the only source of exact value labels for off-path
siblings, and the held-out ranking-gate set.

Key mechanics:

* **Sub-searches start from the stored child states** (schema 2+:
  ``br_child_poses`` / ``br_child_last``), never from re-rolled actions — the
  CUDA GAT forward is non-bit-reproducible, so a regenerated precise child is
  not the state the teacher scored.
* **Obstacle GAT features are reconstructed by seeded replay.**  The GAT
  consumes obstacle ``[x, y, cosθ, sinθ]``; θ is world-random and absent from
  schema-2 shards (schema 3 stores ``obstacle_states`` directly).  For
  schema-2 shards we rebuild the episode's world with ``seed_episode`` +
  ``reset()`` and **fail loudly** unless the replayed obstacle centres/radii
  match the shard (atol 1e-2 — RNG drift moves obstacles by metres, not mm).
* **Consistency gate per node**: the analytic h and terminal/collision flags
  are recomputed from every stored child state and compared to the logged
  branch rows; any disagreement drops the node (guards against stale shards
  or dynamics drift).  Collision is recomputed collision-first, matching the
  search on both pre- and post-sub-step-truncation shards.
* **Plateau-stratified sampling**: half the node budget goes to the
  lowest-f-spread candidates (the BFS-flooding blocking/jiggle states where
  Σd has no gradient — the states sibling ranking exists for), half is
  stratified over depth × n_unreached buckets.

Output: one JSON line per resolved node in ``<out>/resolution_<shard>.jsonl``
with the trace file, node row, and per-child ``(aidx, mode, group, status,
solved, cap_hit, y=plan_cost, expansions)``.  Join back to shards by
``(trace_file, node_row)``.

Usage (workers = disjoint ``--shard-index`` over the same sample):
    python -m robot_nav.collect_sibling_values \
        --traces runs/value_corpus/astar14_coupled/traces \
        --out runs/value_corpus/sibling_resolution \
        --nodes 200 --shard-index 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.eval_gaz14_lazy import build_env, resolve_device
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.switcher_env import seed_episode
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    ForwardModel14,
    ModelState,
    analytic_cost_to_go,
)
from robot_nav.models.MARL.capswitcher.rl.shield import ShieldGeometry
from robot_nav.models.MARL.capswitcher_14.rl.search.best_first import (
    EVALUATIONS,
    BestFirstSearch14,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.trace import load_plan
from robot_nav.SIM_ENV.corridor_layout import CorridorLayout

logger.disable("irsim")

COARSE, PRECISE = 0, 1
H_RTOL = 1e-3          # analytic h recompute tolerance (f32 shard rounding)
OBS_ATOL = 1e-2        # replayed obstacle xy/r must match the shard this tight
                       # (drifted seeding gives O(1) m differences, not mm —
                       # this discriminates same-RNG-path vs different-world)


# ---------------------------------------------------------------------------
# Candidate enumeration and sampling
# ---------------------------------------------------------------------------

def _node_candidates(plan: dict, path: str, goal_threshold: float) -> list[dict]:
    """One record per expanded node with ≥2 searchable children."""
    n_nodes = len(plan["g"])
    per_node = np.bincount(plan["br_node"], minlength=n_nodes)
    base = np.concatenate([[0], np.cumsum(per_node)])
    goals = plan["goals"]
    out = []
    for row in range(n_nodes):
        s = slice(base[row], base[row + 1])
        safe = plan["br_safe"][s]
        cg, ch = plan["br_child_g"][s], plan["br_child_h"][s]
        term = plan["br_child_terminal"][s]
        coll = plan["br_child_collision"][s]
        legal = safe & np.isfinite(cg)
        searchable = legal & ~term & ~coll
        if searchable.sum() < 2:
            continue
        f = (cg + np.where(term, 0.0, ch))[legal & ~coll]
        d = np.linalg.norm(plan["poses"][row][:, :2] - goals, axis=1)
        out.append({
            "file": path,
            "row": row,
            "depth": int(plan["depth"][row]),
            "n_unreached": int((d > goal_threshold).sum()),
            "f_spread": float(f.max() - f.min()) if len(f) else np.inf,
        })
    return out


def _sample_nodes(
    cands: list[dict], n_nodes: int, plateau_frac: float, rng, per_plan: int
) -> list[dict]:
    """Half budget from lowest f-spread, rest stratified depth × n_unreached."""
    # cap per plan so coverage spreads across episodes
    by_plan: dict[str, int] = {}
    kept = []
    order = rng.permutation(len(cands))
    for i in order:
        c = cands[i]
        if by_plan.get(c["file"], 0) < per_plan:
            by_plan[c["file"]] = by_plan.get(c["file"], 0) + 1
            kept.append(c)
    n_plateau = int(round(n_nodes * plateau_frac))
    kept.sort(key=lambda c: c["f_spread"])
    plateau = kept[:n_plateau]
    rest = kept[n_plateau:]
    if not rest:
        return plateau
    # stratify remainder over depth/n_unreached terciles
    depths = np.array([c["depth"] for c in rest])
    unre = np.array([c["n_unreached"] for c in rest])
    db = np.searchsorted(np.quantile(depths, [1 / 3, 2 / 3]), depths)
    ub = np.searchsorted(np.quantile(unre, [1 / 3, 2 / 3]), unre)
    buckets: dict[tuple, list] = {}
    for c, b in zip(rest, zip(db, ub)):
        buckets.setdefault(b, []).append(c)
    for b in buckets.values():
        rng.shuffle(b)
    picked, i = [], 0
    while len(picked) < n_nodes - len(plateau) and any(buckets.values()):
        for b in list(buckets):
            if buckets[b]:
                picked.append(buckets[b].pop())
                if len(picked) >= n_nodes - len(plateau):
                    break
        i += 1
    return plateau + picked


# ---------------------------------------------------------------------------
# World / model reconstruction
# ---------------------------------------------------------------------------

class WorldReplayer:
    """Rebuild per-episode obstacle GAT features by seeded replay."""

    def __init__(self, device, cost, meta):
        layout = None
        if meta.get("world") == "corridor":
            layout = CorridorLayout()   # corpus runs use the CLI defaults
        self.env, self.coarse, self.sim = build_env(
            device, cost=cost, goal_threshold=meta["goal_threshold"],
            backbone_ckpt=meta["backbone_ckpt"], layout=layout,
            precise_config=meta.get("precise_config", "all"),
            coupled=meta.get("coupled", False),
        )
        self._cache: dict[int, np.ndarray] = {}

    def obstacle_states(self, seed: int, plan: dict) -> np.ndarray:
        if seed not in self._cache:
            seed_episode(self.env, seed)
            self.env.reset()
            st = self.sim.get_obstacle_states()
            geom = ShieldGeometry.from_sim(self.sim)
            if not (np.allclose(st[:, :2], plan["obstacle_xy"], atol=OBS_ATOL)
                    and np.allclose(geom.obstacle_r, plan["obstacle_r"],
                                    atol=OBS_ATOL)):
                raise RuntimeError(
                    f"seed {seed}: replayed obstacles do not match the shard "
                    "— reset/seeding drifted since collection; refusing to "
                    "reconstruct GAT features"
                )
            self._cache[seed] = st
        st = self._cache[seed]
        if not np.allclose(st[:, :2], plan["obstacle_xy"], atol=OBS_ATOL):
            raise RuntimeError("shard/seed obstacle mismatch across plans")
        return st


def _leaf(meta):
    scale = float(meta.get("analytic_alpha_scale") or 1.0)
    if scale == 1.0:
        return None
    return lambda m, ms: analytic_cost_to_go(
        m.goal_distances(ms), m.goal_threshold, scale * m.analytic_alpha()
    )


def _make_model(rep: WorldReplayer, meta, plan, obstacle_states) -> ForwardModel14:
    """Fresh model (fresh transition counters) sharing heavy components."""
    return ForwardModel14(
        backbone=rep.env.backbone,
        coarse=rep.coarse,
        goals=plan["goals"].astype(np.float64),
        obstacle_states=obstacle_states,
        geom=ShieldGeometry(
            rho=float(plan["rho"]),
            obstacle_xy=plan["obstacle_xy"].astype(np.float64),
            obstacle_r=plan["obstacle_r"].astype(np.float64),
        ),
        step_time=float(meta.get("step_time", 0.3)),
        selection_interval=int(meta["selection_interval"]),
        lin_max=float(meta.get("lin_max", 0.5)),
        d_safe=float(meta["d_safe"]),
        goal_threshold=float(meta["goal_threshold"]),
        cost=rep.env.cost,
        leaf_value=_leaf(meta),
        coupling=rep.env.coupling,
        precise_groups=rep.env.precise_groups,
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _resolve_node(rep, meta, plan, cand, cap) -> dict | None:
    obstacle_states = rep.obstacle_states(int(plan["seed"]), plan)
    row = cand["row"]
    per_node = np.bincount(plan["br_node"], minlength=len(plan["g"]))
    base = int(np.concatenate([[0], np.cumsum(per_node)])[row])
    n_br = int(per_node[row])

    # consistency gate: recomputed h / flags must match the logged rows
    probe = _make_model(rep, meta, plan, obstacle_states)
    children = []
    for a in range(n_br):
        bi = base + a
        rec = {
            "aidx": a,
            "mode": int(plan["br_mode"][bi]),
            "group": int(plan["br_group"][bi]),
            "pgroup": int(plan["br_pgroup"][bi]),
            "step_cost": float(plan["br_step_cost"][bi]),
        }
        if not plan["br_safe"][bi] or not np.isfinite(plan["br_child_g"][bi]):
            rec["status"] = "refuted"
            children.append(rec)
            continue
        ms = ModelState(
            poses=plan["br_child_poses"][bi].astype(np.float64),
            last_actions=plan["br_child_last"][bi].astype(np.float64),
        )
        if np.isnan(ms.poses).any():
            return None                     # schema violation — drop node
        # Collision-first, mirroring the search: precise rollouts truncate at
        # the first colliding sub-step, so a mid-rollout collision's stored
        # end state is itself colliding — an endpoint recompute reproduces the
        # flag without re-rolling the GAT (which is not bit-reproducible).
        # Endpoint-collision children of pre-truncation shards satisfy the
        # same rule, so the gate accepts both schemas.
        coll = bool(probe.collision_pred(ms))
        term = bool((not coll) and probe.all_reached(ms))
        if term != bool(plan["br_child_terminal"][bi]) or \
                coll != bool(plan["br_child_collision"][bi]):
            return None                     # gate: flag mismatch
        if coll:
            rec["status"] = "collision"
        elif term:
            rec.update(status="terminal", solved=True, y=0.0)
        else:
            h = float(probe.cost_to_go(ms))
            logged = float(plan["br_child_h"][bi])
            if abs(h - logged) > H_RTOL * max(1.0, abs(logged)):
                return None                 # gate: h mismatch
            rec.update(status="searchable", _ms=ms)
        children.append(rec)

    search = BestFirstSearch14(EVALUATIONS["astar"], max_transitions=cap)
    for rec in children:
        if rec.get("status") != "searchable":
            continue
        ms = rec.pop("_ms")
        model = _make_model(rep, meta, plan, obstacle_states)
        with torch.no_grad():
            res = search.run(model, ms)
        rec.update(
            solved=bool(res.solved), cap_hit=bool(res.cap_hit),
            y=float(res.plan_cost), expansions=int(res.expansions),
            transitions=int(model.n_coarse_vets + model.n_precise_expansions),
        )
    return {
        "file": cand["file"], "row": row, "depth": cand["depth"],
        "n_unreached": cand["n_unreached"], "f_spread": cand["f_spread"],
        "episode": int(plan["episode"]), "seed": int(plan["seed"]),
        "decision_index": int(plan["decision_index"]),
        "children": children,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True,
                    help="trace root: contains <algo>_<config>_s<seed>/ dirs")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--nodes", type=int, default=200,
                    help="total nodes to resolve (across all shards)")
    ap.add_argument("--plateau-frac", type=float, default=0.5)
    ap.add_argument("--per-plan", type=int, default=2,
                    help="max sampled nodes per planning call")
    ap.add_argument("--max-transitions", type=int, default=0,
                    help="sub-search cap; 0 = the teacher cap from meta")
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    roots = sorted(p for p in Path(args.traces).iterdir() if p.is_dir())
    if not roots:
        raise SystemExit(f"no shard dirs under {args.traces}")
    meta = json.loads((roots[0] / "meta.json").read_text())
    if meta.get("value_model"):
        raise SystemExit("learned-h teacher traces are out of scope")
    cap = args.max_transitions or int(meta["max_transitions"])
    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(meta.get(
        "cost_config", "robot_nav/models/MARL/capswitcher_14/cost_14robots.yaml"))
    assert abs(cost.precise_unit - meta["precise_unit"]) < 1e-9, \
        "cost YAML drifted from the shard meta"

    files = sorted(glob.glob(str(Path(args.traces) / "*" / "plan_*.npz")))
    print(f"{len(files)} plans; enumerating candidates ...")
    cands = []
    for f in files:
        plan = load_plan(f)
        if "br_child_poses" not in plan:
            continue                        # schema-1 shard: cannot resolve
        cands.extend(_node_candidates(plan, f, float(meta["goal_threshold"])))
    print(f"{len(cands)} candidate nodes")
    rng = np.random.default_rng(args.sample_seed)
    sampled = _sample_nodes(cands, args.nodes, args.plateau_frac, rng,
                            args.per_plan)
    mine = [c for i, c in enumerate(sampled)
            if i % args.num_shards == args.shard_index]
    print(f"sampled {len(sampled)} nodes; this shard resolves {len(mine)}")

    rep = WorldReplayer(device, cost, meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"resolution_{args.shard_index:02d}.jsonl"
    n_done = n_dropped = 0
    with open(out_path, "a") as fh:
        for cand in mine:
            plan = load_plan(cand["file"])
            rec = _resolve_node(rep, meta, plan, cand, cap)
            if rec is None:
                n_dropped += 1
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += 1
            print(f"[{n_done}/{len(mine)}] row {rec['row']} "
                  f"seed {rec['seed']} d{rec['decision_index']} "
                  f"f_spread {rec['f_spread']:.1f}")
    print(f"done: {n_done} resolved, {n_dropped} dropped (gate), -> {out_path}")


if __name__ == "__main__":
    main()
