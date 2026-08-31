"""
Watch an executed A* trajectory straight from saved trace shards, with the
per-decision child f-spread printed and overlaid — the "where along the path
does the search go blind?" viewer.

Unlike ``render_gaz14`` this does not re-run the sim or load any checkpoint:
the shards store every expanded node's pose and every child's ``g``/``h``, so
the executed path (root -> goal/best node per planning call, concatenated
across replans) is reconstructed exactly from disk.  Frames are the stored
decision-boundary states (~1 per selection interval), not sub-steps.

Per decision the console prints and the figure overlays:

* ``fs_all``    — max-min of child f over legal non-collision children (the
  sampler's ``f_spread``);
* ``fs_coarse`` — same restricted to coarse children (can the search rank its
  cheap options?);
* ``h_spr``     — max-min of child h alone (pure heuristic flatness);
* ``c_spr``     — max-min of child step cost (the pricing part of f);
* ``nC/nP``     — searchable coarse / precise children counts.

A node is flagged FLAT when ``fs_all <= --flat-thr``.  Comparing ``fs_all``
with ``h_spr``/``c_spr`` on screen shows directly whether a large spread is
heuristic signal or just the coarse/precise price gap.

Usage:
    # list episodes available in a shard
    python -m robot_nav.render_trace_fspread \
        --shard runs/value_corpus/astar14_coupled/traces/astar_all_s2010 --list

    # watch one episode live (display required; ~2 decisions/s)
    python -m robot_nav.render_trace_fspread \
        --shard runs/value_corpus/astar14_coupled/traces/astar_all_s2010 \
        --episode 17

    # headless: write a gif instead of opening a window
    python -m robot_nav.render_trace_fspread --shard <dir> --episode 17 \
        --save /tmp/ep17.gif
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from robot_nav.models.MARL.capswitcher_14.rl.search.trace import load_plan

COARSE, PRECISE = 0, 1


# ---------------------------------------------------------------------------
# Path reconstruction and per-node spread decomposition
# ---------------------------------------------------------------------------

def _branch_base(plan: dict) -> np.ndarray:
    per_node = np.bincount(plan["br_node"], minlength=len(plan["g"]))
    return np.concatenate([[0], np.cumsum(per_node)]).astype(int)


def node_spreads(plan: dict, row: int, base: np.ndarray) -> dict:
    """Spread decomposition of one expanded node's children."""
    s = slice(base[row], base[row + 1])
    safe = plan["br_safe"][s]
    cg, ch = plan["br_child_g"][s], plan["br_child_h"][s]
    term, coll = plan["br_child_terminal"][s], plan["br_child_collision"][s]
    mode, cost = plan["br_mode"][s], plan["br_step_cost"][s]
    legal = safe & np.isfinite(cg) & ~coll
    f = cg + np.where(term, 0.0, ch)

    def spread(mask, vals):
        v = vals[mask]
        return float(v.max() - v.min()) if len(v) >= 2 else float("nan")

    searchable = legal & ~term
    return {
        "fs_all": spread(legal, f),
        "fs_coarse": spread(legal & (mode == COARSE), f),
        "h_spr": spread(searchable, ch),
        "c_spr": spread(legal, cost),
        "nC": int((searchable & (mode == COARSE)).sum()),
        "nP": int((searchable & (mode == PRECISE)).sum()),
    }


def executed_path(plan: dict) -> list[dict]:
    """
    The plan's executed decisions in order: one record per on-path expanded
    node (state, spreads, chosen branch), plus the final child state.
    """
    base = _branch_base(plan)
    rows: list[int] = []
    r = int(plan["goal_parent_row"])
    while r != -1:
        rows.append(r)
        r = int(plan["parent_row"][r])
    rows.reverse()

    if not rows:
        # Empty plan: cap hit with the root as best node.  The switcher
        # executes one live precise fallback from the root state; show that
        # state with the root's spread decomposition (what the search saw
        # when it failed to rank anything).
        root = int(np.where(plan["parent_row"] == -1)[0][0])
        return [{
            "row": root,
            "poses": plan["poses"][root],
            "mode": PRECISE,
            "group": -1,
            "pgroup": -1,
            "step_cost": float("nan"),
            "next_poses": plan["poses"][root],
            "fallback": True,
            **node_spreads(plan, root, base),
        }]

    out = []
    for i, row in enumerate(rows):
        aidx = (int(plan["parent_aidx"][rows[i + 1]]) if i + 1 < len(rows)
                else int(plan["goal_aidx"]))
        bi = base[row] + aidx
        out.append({
            "row": row,
            "poses": plan["poses"][row],
            "mode": int(plan["br_mode"][bi]),
            "group": int(plan["br_group"][bi]),
            "pgroup": int(plan["br_pgroup"][bi]),
            "step_cost": float(plan["br_step_cost"][bi]),
            "next_poses": plan["br_child_poses"][bi],
            **node_spreads(plan, row, base),
        })
    return out


def episode_plans(shard: Path, episode: int) -> list[str]:
    files = glob.glob(str(shard / f"plan_ep{episode:04d}_d*.npz"))
    return sorted(files, key=lambda f: int(re.search(r"_d(\d+)\.npz$", f).group(1)))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(shard: Path, episode: int, flat_thr: float, fps: float,
           save: str | None) -> None:
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plans = episode_plans(shard, episode)
    if not plans:
        raise SystemExit(f"no plans for episode {episode} in {shard}")

    # concatenate executed paths over replans
    steps: list[dict] = []
    for f in plans:
        plan = load_plan(f)
        path = executed_path(plan)
        for j, rec in enumerate(path):
            rec["dec"] = int(plan["decision_index"])
            rec["replan"] = j == 0
            rec["cap_hit"] = bool(plan["cap_hit"])
            rec["solved"] = bool(plan["solved"])
        steps.extend(path)
    first = load_plan(plans[0])
    goals, rho = first["goals"], float(first["rho"])
    oxy, orr = first["obstacle_xy"], first["obstacle_r"]
    n = goals.shape[0]

    print(f"episode {episode} (seed {int(first['seed'])}): "
          f"{len(plans)} plans, {len(steps)} executed decisions")
    hdr = (f"{'k':>3} {'dec':>3} {'mode':>7} {'fs_all':>8} {'fs_crs':>8} "
           f"{'h_spr':>8} {'c_spr':>8} {'nC':>3} {'nP':>3}  flags")
    print(hdr)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_aspect("equal")
    all_xy = np.concatenate(
        [goals, oxy] + [rec["poses"][:, :2] for rec in steps]
        if len(oxy) else [goals] + [rec["poses"][:, :2] for rec in steps]
    )
    lo, hi = all_xy.min(axis=0) - 2, all_xy.max(axis=0) + 2
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1])
    for xy, r in zip(oxy, orr):
        ax.add_patch(Circle(xy, r, color="#c3c2b7", zorder=1))
    ax.scatter(goals[:, 0], goals[:, 1], marker="x", c="#52514e", s=40, zorder=2)
    bodies = [ax.add_patch(Circle((0, 0), rho, color="#2a78d6", zorder=3))
              for _ in range(n)]
    ticks = [ax.plot([], [], color="#fcfcfb", linewidth=1.5, zorder=4)[0]
             for _ in range(n)]
    title = ax.set_title("", loc="left", fontsize=10)
    banner = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=13,
                     color="#eb6834", fontweight="bold")

    def draw(k: int):
        rec = steps[k]
        arrived = np.linalg.norm(rec["poses"][:, :2] - goals, axis=1) <= 0.3
        for i in range(n):
            x, y, th = rec["poses"][i]
            bodies[i].center = (x, y)
            bodies[i].set_color("#1baf7a" if arrived[i] else "#2a78d6")
            ticks[i].set_data([x, x + rho * np.cos(th)],
                              [y, y + rho * np.sin(th)])
        flat = np.isfinite(rec["fs_all"]) and rec["fs_all"] <= flat_thr
        mode = ("fallbk" if rec.get("fallback") else
                "coarse" if rec["mode"] == COARSE else
                f"prec{'' if rec['pgroup'] < 0 else rec['pgroup']}")
        title.set_text(
            f"ep {episode}  dec {rec['dec']}  step {k}/{len(steps) - 1}  "
            f"{mode}  fs_all={rec['fs_all']:.1f}  h_spr={rec['h_spr']:.1f}  "
            f"c_spr={rec['c_spr']:.1f}"
        )
        banner.set_text("FLAT — search has no ranking here" if flat else "")
        flags = " ".join(w for w, on in (
            ("REPLAN", rec["replan"] and k > 0),
            ("cap-hit", rec["replan"] and rec["cap_hit"]),
            ("FLAT", flat)) if on)
        print(f"{k:>3} {rec['dec']:>3} {mode:>7} {rec['fs_all']:>8.2f} "
              f"{rec['fs_coarse']:>8.2f} {rec['h_spr']:>8.2f} "
              f"{rec['c_spr']:>8.2f} {rec['nC']:>3} {rec['nP']:>3}  {flags}")
        return bodies + ticks + [title, banner]

    if save:
        from matplotlib.animation import FuncAnimation, PillowWriter
        anim = FuncAnimation(fig, draw, frames=len(steps), blit=False)
        anim.save(save, writer=PillowWriter(fps=fps))
        print(f"saved -> {save}")
    else:
        plt.ion()
        plt.show()
        for k in range(len(steps)):
            draw(k)
            fig.canvas.draw_idle()
            plt.pause(1.0 / fps)
        plt.ioff()
        plt.show()      # keep the final frame up until closed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, required=True,
                    help="one shard dir, e.g. .../traces/astar_all_s2010")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--list", action="store_true",
                    help="list episodes in the shard and exit")
    ap.add_argument("--flat-thr", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--save", type=str, default=None,
                    help="write a gif instead of opening a window")
    args = ap.parse_args()

    shard = Path(args.shard)
    if args.list or args.episode is None:
        eps = sorted({int(re.search(r"plan_ep(\d+)_", f).group(1))
                      for f in glob.glob(str(shard / "plan_*.npz"))})
        print(f"{len(eps)} episodes in {shard.name}: {eps}")
        if args.episode is None and not args.list:
            print("pass --episode <n> to render one")
        return
    render(shard, args.episode, args.flat_thr, args.fps, args.save)


if __name__ == "__main__":
    main()
