"""
CoarseSteering: least-squares coarse group steering over a rank-deficient
actuation matrix.

Given desired per-robot headings (toward their goals), solves:

    t* = pinv(A_reduced) @ dθ_desired
    dθ_actual = A_reduced @ t*

This produces collective rotation that best approximates goal-oriented
headings within the reachable subspace of A_reduced.

Physical interpretation
-----------------------
The membership matrix M (6 robots × 4 groups) defines which robots belong
to each group.  The actuation matrix A is the complement of M: A[i,s] = 1
means robot i rotates when group s is activated (i.e. robot i is NOT a
member of group s).

Full A (6 robots × 4 groups):
    Robot 1: [1, 1, 0, 1]
    Robot 2: [1, 0, 1, 1]
    Robot 3: [1, 0, 0, 1]
    Robot 4: [0, 1, 1, 1]
    Robot 5: [0, 1, 0, 1]
    Robot 6: [0, 0, 1, 1]

Rank-2 constraint: use only the first 2 columns (groups 1 and 2).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Full actuation matrix (6 robots × 4 groups).
# Constructed as complement of the membership matrix M shown in the project
# summary.
# ---------------------------------------------------------------------------
_A_FULL_6 = np.array(
    [
        [1, 1, 0, 1],  # Robot 0
        [1, 0, 1, 1],  # Robot 1
        [1, 0, 0, 1],  # Robot 2
        [0, 1, 1, 1],  # Robot 3
        [0, 1, 0, 1],  # Robot 4
        [0, 0, 1, 1],  # Robot 5
    ],
    dtype=np.float64,
)


def _angle_wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into (−π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class CoarseSteering:
    """
    Least-squares coarse group steering for a swarm of coupled unicycle robots.

    Uses the rank-2 actuation matrix (first ``use_groups`` columns of A) to
    produce per-robot angular velocity commands that best steer the swarm
    toward each robot's individual goal in the least-squares sense.

    Args:
        num_robots: Number of robots (must be 6 for the default matrix).
        use_groups: Number of actuation groups (rank of A_reduced). Default 2.
        lin_vel:    Fixed forward linear velocity for all robots in coarse mode,
                    in sim-input format (passed directly to the simulator).
                    Corresponds to the fraction of max speed. Default 0.25.
    """

    def __init__(
        self,
        num_robots: int = 6,
        use_groups: int = 2,
        lin_vel: float = 0.25,
    ) -> None:
        if num_robots != 6:
            raise ValueError(
                "CoarseSteering currently only supports num_robots=6 "
                f"(got {num_robots})."
            )
        self.num_robots = num_robots
        self.lin_vel = lin_vel

        # Build reduced actuation matrix (N, K) and its Moore-Penrose
        # pseudoinverse (K, N).
        self.A_reduced: np.ndarray = _A_FULL_6[:, :use_groups].copy()
        self.pinv_A: np.ndarray = np.linalg.pinv(self.A_reduced)

    # ------------------------------------------------------------------

    def compute_actions(self, robot_state: np.ndarray) -> list[list[float]]:
        """
        Compute coarse sim-input actions for all robots.

        Computes the per-robot desired heading error (dθ_desired), projects it
        onto the reachable rotation subspace via the pseudoinverse of A_reduced,
        and converts the result to angular velocity commands.

        Args:
            robot_state: Per-robot state array of shape (N, 11).
                Column layout (same as TD3Obstacle.prepare_state output):
                    0, 1  : px, py       — robot position
                    2, 3  : cos θ, sin θ — robot heading
                    4     : dist_to_goal
                    5, 6  : cos_err, sin_err
                    7, 8  : lin_vel_obs, ang_vel_obs
                    9, 10 : gx, gy       — goal position

        Returns:
            List of N ``[lin_vel, ang_vel]`` pairs in sim-input format:
                lin_vel  ∈ [0, 0.5] (passed directly; not actor-space)
                ang_vel  ∈ [−1, 1]
        """
        s = np.asarray(robot_state, dtype=np.float64)  # (N, 11)

        px, py   = s[:, 0], s[:, 1]
        cos_h, sin_h = s[:, 2], s[:, 3]
        gx, gy   = s[:, 9], s[:, 10]

        theta_current = np.arctan2(sin_h, cos_h)             # (N,)
        theta_desired = np.arctan2(gy - py, gx - px)         # (N,)
        d_theta = _angle_wrap(theta_desired - theta_current)  # (N,)

        # Least-squares rotation:  t* = pinv(A) @ dθ  →  dθ_actual = A @ t*
        t_star       = self.pinv_A @ d_theta             # (K,)
        d_theta_act  = self.A_reduced @ t_star            # (N,)

        # Map to angular velocity ∈ [−1, 1]  (π rad ≡ full angular velocity)
        ang_vels = np.clip(d_theta_act / np.pi, -1.0, 1.0)  # (N,)

        return [[self.lin_vel, float(ang_vels[i])] for i in range(self.num_robots)]
