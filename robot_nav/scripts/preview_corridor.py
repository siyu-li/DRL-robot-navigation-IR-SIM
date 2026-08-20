"""
Draw sample corridor episodes without booting irsim.

Tuning band widths against the live simulator is slow and needs the GPU box;
the layout is pure geometry, so it can be inspected here instead.  Goals are
drawn uniformly in the goal band, which is what irsim's ``set_random_goal``
does apart from its obstacle-rejection step — close enough to judge whether a
band setting gives the sparse → dense → sparse profile you want.

Usage:
    python -m robot_nav.scripts.preview_corridor --episodes 3 --out preview.png
    python -m robot_nav.scripts.preview_corridor --band 0.45 0.55 --bidirectional
"""

from __future__ import annotations

import argparse
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robot_nav.SIM_ENV.corridor_layout import (  # noqa: E402
    LEFT_TO_RIGHT,
    CorridorLayout,
    sample_obstacles,
    sample_starts,
)

X_RANGE = (0.0, 15.0)
Y_RANGE = (0.0, 15.0)


def draw(ax, layout: CorridorLayout, seed: int, n_robots: int, n_obstacles: int,
         obstacle_radius: float, robot_radius: float) -> float:
    """Draw one episode; return the tightest obstacle-pair gap in metres."""
    radii = np.full(n_obstacles, obstacle_radius)
    np.random.seed(seed)
    obstacles = sample_obstacles(radii, X_RANGE, Y_RANGE, layout)
    random.seed(seed)
    starts = sample_starts(
        n_robots, X_RANGE, Y_RANGE, layout, obstacles[:, :2], radii,
        robot_radius=robot_radius,
    )
    directions = layout.directions(n_robots)

    y0, y1 = layout.y_band(Y_RANGE)
    if not layout.obstacle_free:
        bx0, bx1 = layout.obstacle_band(X_RANGE)
        ax.axvspan(bx0, bx1, color="saddlebrown", alpha=0.10, lw=0)
    for d, colour in ((LEFT_TO_RIGHT, "tab:blue"), (-LEFT_TO_RIGHT, "tab:red")):
        if not np.any(directions == d):
            continue
        sx0, sx1 = layout.start_band(X_RANGE, d)
        gx0, gx1 = layout.goal_band(X_RANGE, d)
        ax.axvspan(sx0, sx1, color=colour, alpha=0.07, lw=0)
        ax.axvspan(gx0, gx1, color=colour, alpha=0.04, lw=0)

    for (x, y, _), r in zip(obstacles, radii):
        ax.add_patch(plt.Circle((x, y), r, color="saddlebrown", alpha=0.85))

    goals = np.zeros((n_robots, 2))
    for i, d in enumerate(directions):
        gx0, gx1 = layout.goal_band(X_RANGE, int(d))
        gy = starts[i, 1] if layout.aligned_goals else np.random.uniform(y0, y1)
        goals[i] = (np.random.uniform(gx0, gx1), gy)

    for i, d in enumerate(directions):
        colour = "tab:blue" if d == LEFT_TO_RIGHT else "tab:red"
        ax.add_patch(plt.Circle(starts[i, :2], robot_radius, color=colour))
        ax.plot(*goals[i], marker="x", color=colour, ms=6, mew=1.6)
        ax.annotate(
            "", xy=goals[i], xytext=starts[i, :2],
            arrowprops=dict(arrowstyle="->", color=colour, alpha=0.25, lw=0.9),
        )

    d = np.linalg.norm(obstacles[:, None, :2] - obstacles[None, :, :2], axis=2)
    np.fill_diagonal(d, np.inf)
    gap = float(np.min(d - radii[:, None] - radii[None, :]))

    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.set_title(
        f"seed {seed}   " + (
            "obstacles parked" if layout.obstacle_free
            else f"tightest gate gap {gap:.2f} m"
        ),
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    return gap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default="corridor_preview.png")
    ap.add_argument("--robots", type=int, default=14)
    ap.add_argument("--obstacles", type=int, default=7)
    ap.add_argument("--obstacle-radius", type=float, default=0.7)
    ap.add_argument("--robot-radius", type=float, default=0.2)
    # --- layout knobs, mirroring eval_gaz14_lazy.add_layout_args ---
    ap.add_argument("--band", type=float, nargs=2, default=(0.40, 0.60))
    ap.add_argument("--start-width", type=float, default=0.25)
    ap.add_argument("--goal-width", type=float, default=0.25)
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--empty", action="store_true",
                    help="sanity mode: obstacles parked along the top wall")
    ap.add_argument("--aligned", action="store_true",
                    help="sanity mode: y-lanes, goal y pinned to start y")
    args = ap.parse_args()

    layout = CorridorLayout(
        band=(args.band[0], args.band[1]),
        start_width=args.start_width,
        goal_width=args.goal_width,
        min_gap=args.min_gap,
        bidirectional=args.bidirectional,
        obstacle_free=args.empty,
        aligned_goals=args.aligned,
    )
    layout.validate()

    fig, axes = plt.subplots(
        1, args.episodes, figsize=(4.2 * args.episodes, 4.4), squeeze=False
    )
    gaps = [
        draw(axes[0][k], layout, args.seed + k, args.robots, args.obstacles,
             args.obstacle_radius, args.robot_radius)
        for k in range(args.episodes)
    ]
    fig.suptitle(
        f"corridor layout — band {layout.band}, min_gap {layout.min_gap} m"
        f"{', bidirectional' if layout.bidirectional else ''}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}   gate gaps: "
          f"{', '.join(f'{g:.2f}' for g in gaps)} m")


if __name__ == "__main__":
    main()
