"""
Base policies (priors) over action edges for the Gumbel AlphaZero switcher.

A prior is any callable ``prior(model, ms, branches) -> logits`` (one logit per
:class:`Branch`, same order).  It must be *expansion-free*: everything it uses
has to be available the moment a node is expanded, without evaluating children
or the value net — otherwise the prior would spend the very budget it exists to
save.  That is why :class:`HeuristicPrior` is built purely from shield
statistics: coarse vetting (clearance / progress per group) is mandatory anyway
for the frames/safety contract, so those features are free.

Stage 2 (later) is a small learned prior behind the same signature, trained on
the improved root policies ``pi'`` logged by the Gumbel search — see
``GumbelSwitcher`` — via a supervised KL fit (no search-code change needed).
"""

from __future__ import annotations

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE


class HeuristicPrior:
    """
    Hand-designed base policy over expanded branches, from shield statistics.

    For each shield-safe coarse branch ``g``::

        logit(g) = w_progress * progress_g / step_cost_g          # return on motion
                 + w_clearance * min(clearance_g - d_safe, clearance_cap)

    and for the always-present precise branch::

        logit(precise) = precise_bias + w_congestion * (-best_coarse_ratio)

    Reasoning:

    * ``progress / step_cost`` is the myopic "metres of summed goal-distance
      reduction bought per unit of executed motion cost" — the unitless myopic
      version of the quantity the switcher minimises.  Dominant term.
    * The capped clearance margin softly prefers moves that are not scraping
      ``d_safe`` (their successors are more likely to be shield-blocked); the
      cap stops wide-open space from drowning the efficiency term.
    * Precise should look *relatively* better the worse the best coarse option
      is — ``best_coarse_ratio`` is the max efficiency ratio over safe groups
      (0 when none) — plus a global bias, the main coarse/precise tuning knob.

    ``temperature`` divides the logits: high = flatter prior (safer when the
    heuristic is crude; Gumbel's policy-improvement step is robust to it).

    Args:
        w_progress:    Weight of the efficiency ratio (default 3.0).
        w_clearance:   Weight of the capped clearance margin (default 1.0).
        clearance_cap: Cap (m) on the clearance margin (default 0.5).
        precise_bias:  Additive logit for the precise edge (default 0.0).
        w_congestion:  Weight of the anti-correlation with the best coarse
                       ratio on the precise logit (default 1.0).
        temperature:   Softmax temperature dividing all logits (default 1.0).
        d_safe:        Fallback safety margin when the model exposes none.
    """

    def __init__(
        self,
        w_progress: float = 3.0,
        w_clearance: float = 1.0,
        clearance_cap: float = 0.5,
        precise_bias: float = 0.0,
        w_congestion: float = 1.0,
        temperature: float = 1.0,
        d_safe: float = 0.3,
    ) -> None:
        self.w_progress = float(w_progress)
        self.w_clearance = float(w_clearance)
        self.clearance_cap = float(clearance_cap)
        self.precise_bias = float(precise_bias)
        self.w_congestion = float(w_congestion)
        self.temperature = float(temperature)
        self.d_safe = float(d_safe)

    def __call__(self, model, ms, branches) -> np.ndarray:
        d_safe = float(getattr(model, "d_safe", self.d_safe))
        logits = np.zeros(len(branches), dtype=np.float64)

        ratios = []
        for i, b in enumerate(branches):
            if b.mode != COARSE:
                continue
            ratio = float(b.candidate.progress) / max(float(b.step_cost), 1e-9)
            margin = min(float(b.candidate.clearance) - d_safe, self.clearance_cap)
            logits[i] = self.w_progress * ratio + self.w_clearance * margin
            ratios.append(ratio)

        best_ratio = max(ratios) if ratios else 0.0
        for i, b in enumerate(branches):
            if b.mode != COARSE:
                logits[i] = self.precise_bias - self.w_congestion * best_ratio

        return logits / max(self.temperature, 1e-9)


class UniformPrior:
    """Flat prior — every legal edge equally likely (baseline / tests)."""

    def __call__(self, model, ms, branches) -> np.ndarray:
        return np.zeros(len(branches), dtype=np.float64)
