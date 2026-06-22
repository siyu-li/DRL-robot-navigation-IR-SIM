"""
Shielded switcher policies P0 and P1 (no learning).

Both sit on top of the same hard safety shield (``shield.py``): every coarse
group is vetted by lookahead, and only verified-safe coarse moves are eligible.
``precise`` is always available as the safe fallback, so a decision is made at
every state.

  * **P0 — "coarse whenever safe."**  If any group is safe, take coarse (the
    safe group with the largest clearance); otherwise precise.  Tests the
    hypothesis that, with safety handled, the trade-off collapses to "always
    coarse when you can."

  * **P1 — "coarse when safe AND productive."**  Among the safe groups, take the
    one with the largest predicted goal progress, but only if that progress
    clears ``progress_threshold``; otherwise precise.  Adds the steerability
    axis that P0 ignores — a safe coarse move that does not advance the swarm is
    declined in favour of the steerable (but costly) precise mode.

The gap P0 → P1 isolates how much the progress/steerability axis matters; a
later P2 (short lookahead / RL over the same safe set) would reveal any
*sequential* efficiency the myopic rules miss.

A decision is returned as a plain dict so the eval harness can drive
``SwitcherEnv.step`` directly:

    {"mode": 0|1, "group": int|None, "frames": list|None, "candidates": [...]}

For coarse, ``frames`` are the exact vetted frames — they must be executed
as-is (re-generating would draw a different random column and a different path).
"""

from __future__ import annotations

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.shield import (
    CoarseCandidate,
    ShieldGeometry,
    evaluate_coarse_group,
)

COARSE: int = 0
PRECISE: int = 1


class ShieldedSwitcher:
    """
    Rule-based shielded switcher (P0 or P1).

    Args:
        coarse:             CoarseSteering primitive (shared with the env).
        sim:                MARL_SIM_OBSTACLE (for live collision geometry).
        mode:               "P0" or "P1".
        d_safe:             Clearance margin (m) a coarse move must keep to be
                            eligible.  The shield's only conservatism knob.
        progress_threshold: P1 only — minimum predicted goal-distance reduction
                            (m, summed over members) for a coarse move to be
                            preferred over precise.
        default_rho:        Fallback robot radius if the sim does not expose one.
    """

    def __init__(
        self,
        coarse,
        sim,
        mode: str = "P1",
        d_safe: float = 0.3,
        progress_threshold: float = 0.05,
        default_rho: float = 0.2,
    ) -> None:
        if mode not in ("P0", "P1"):
            raise ValueError(f"mode must be 'P0' or 'P1' (got {mode!r}).")
        self.coarse = coarse
        self.sim = sim
        self.mode = mode
        self.d_safe = d_safe
        self.progress_threshold = progress_threshold
        self.default_rho = default_rho

    def candidates(self, robot_state: np.ndarray) -> list[CoarseCandidate]:
        """Vet every selectable coarse group at the current state."""
        geom = ShieldGeometry.from_sim(self.sim, default_rho=self.default_rho)
        return [
            evaluate_coarse_group(self.coarse, robot_state, g, geom, self.d_safe)
            for g in self.coarse.selectable_groups()
        ]

    def decide(self, robot_state: np.ndarray) -> dict:
        """Return the shielded decision for the current state."""
        cands = self.candidates(robot_state)
        safe = [c for c in cands if c.safe]

        chosen: CoarseCandidate | None = None
        if safe:
            if self.mode == "P0":
                # Coarse whenever safe: most-clear safe group.
                chosen = max(safe, key=lambda c: c.clearance)
            else:  # P1: coarse only when safe AND productive enough.
                best = max(safe, key=lambda c: c.progress)
                if best.progress >= self.progress_threshold:
                    chosen = best

        if chosen is None:
            return {"mode": PRECISE, "group": None, "frames": None,
                    "candidates": cands}
        return {"mode": COARSE, "group": chosen.group, "frames": chosen.frames,
                "candidates": cands}
