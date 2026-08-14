"""
Base policies for the **eager** 14-robot search (GAZ14-E).

Different contract from ``priors.py``.  The lazy prior is called on bare stubs
and is forbidden to touch vetting output, because the vet is exactly the cost
it exists to avoid.  Here the eager expansion has already vetted all 22 coarse
groups and rolled out precise before the prior runs, so the shield statistics
are *sunk cost* and the prior is free to use them::

    prior(model, ms, branches) -> logits          # branches carry candidate/progress

Refuted coarse branches are still passed in (fixed 23-wide layout); they are
masked out of every softmax by the node's ``legal`` array, and get logit 0 here
so the logged ``prior_logits`` vector stays finite.

Where these logits actually bite
--------------------------------
At a node with all 23 Q values known, ``σ(q)`` reaches ``c_visit ≈ 50`` while a
sane prior spreads over a few units — so the prior does **not** dominate
scoring, and it should not: the search has real evidence.  It is decisive in
exactly two places:

* **Gumbel top-m root sampling**, scored on ``gum(a) + logits(a)`` alone before
  a single halving round.  Against unit-scale Gumbel noise, a ±3 logit spread
  strongly shapes which arms get any budget at all — and with a budget of a few
  ``expansion_cost`` units, that choice is most of the search.
* The interior ``π′`` tie-break between edges of comparable Q.

That is what fixes the logit scale below: large enough to steer arm sampling,
small enough not to argue with measured Q.
"""

from __future__ import annotations

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE

# Consistency factor making the median absolute deviation an unbiased estimator
# of σ for normal data.
_MAD_TO_SIGMA = 1.4826


class HeuristicPrior14:
    """
    Hand-designed base policy over eagerly expanded branches.

    Two signals, both already paid for by the expansion:

    * ``eff = progress / step_cost`` — metres of summed goal-distance reduction
      bought per unit of executed motion cost.  The myopic, unitless version of
      the quantity the switcher minimises, and the dominant term.  Because
      ``tree_eager`` measures the precise rollout's own progress, the precise
      edge is scored in the *same* currency as the coarse ones — no
      ``precise_bias`` guesswork is needed to balance the two (a small bias
      knob survives for tuning, defaulting to 0).
    * ``clearance − d_safe`` — the capped margin over the shield boundary,
      preferring moves that are not scraping it (their successors are likelier
      to still have safe options).

    Robust standardisation (the part the cost table forces)
    ------------------------------------------------------
    ``cost_14robots.yaml`` prices the four size-7 groups at 8.8 and the
    eighteen size-3/4 groups at 52.7, so ``eff`` is spread *multiplicatively*
    and **bimodally**: on a representative state it runs 0.002 → 0.05 for the
    size-3/4 cluster and 0.24 → 0.50 for the size-7 cluster, with precise down
    near 0.008.  Nearly two decades, in two lumps.

    That breaks both naive treatments.  A plain z-score lets the size-7 cluster
    inflate σ until the eighteen others are indistinguishable.  Standardising
    the *raw* ratio robustly (median/MAD) fixes that end but not the other: the
    MAD is then set by the dense size-3/4 cluster, so the size-7 groups land
    30–60 MAD units out, deep in any saturation's flat tail, and the four edges
    the search most wants ranked collapse onto one value (measured: they
    separated by 2e-4, against unit-scale Gumbel noise).

    So the ratio is compressed **before** it is standardised::

        u = arcsinh(eff / s),  s = median |eff|      # log-like tail, signed
        z = z_clip · tanh((u − median u) / (1.4826 · MAD(u) + ε) / z_clip)

    ``arcsinh`` turns the multiplicative spread into an additive one — it is
    logarithmic for ``|eff| ≫ s`` and linear near zero, so unlike ``log`` it
    handles the negative ``eff`` of a group that would push its members *away*
    from their goals.  The two clusters then land a few units apart with ~1
    unit of internal spread, and MAD standardisation resolves both at once.

    ``tanh`` bounds the result without a hard clip's ordering loss: it is ≈
    identity near the median and strictly monotone everywhere.

    Residual behaviour, measured over random states on the real table: the
    eighteen size-3/4 groups spread cleanly over roughly [−2.4, +1.3], the
    size-7 cluster sits near ``+z_clip``, and separation *within* that cluster
    ranges from ~0.08 to ~3 logits depending on how spread its members' ``eff``
    happen to be — the tight cases still saturate.  That is acceptable **for
    this variant specifically**: the prior's job here is deciding which arms
    enter Gumbel-top-m, and the cluster clears the next-best edge by ~1.7
    logits, so all of it is sampled either way.  Ranking *within* the cluster
    is then done by exact Q, which the eager expansion has already bought.
    Raise ``z_clip`` if a use appears that needs the tail resolved.

    Both transforms are scale-free (``eff / s``, then median/MAD), so the
    logits are invariant to a global rescaling of the cost table — retuning
    prices does not silently retune the prior.

    The clearance term is **centred over the legal coarse edges**, so the
    precise edge — which is never shield-vetted, and about whose safety this
    prior should claim nothing — contributes exactly the average and is neither
    rewarded nor punished for the omission.

    Args:
        w_eff:         Weight of the standardised efficiency score.
        w_clear:       Weight of the centred clearance margin.
        clearance_cap: Cap (m) on the clearance margin — beyond it, more open
                       space carries no extra information.
        precise_bias:  Additive logit on the precise edge; the coarse/precise
                       tuning knob, 0 by default because ``eff`` already prices
                       the trade.
        temperature:   Divides all logits (high = flatter prior).
        z_clip:        Soft saturation level of the standardised efficiency
                       score (approached, never exceeded).
        d_safe:        Fallback safety margin when the model exposes none.
    """

    def __init__(
        self,
        w_eff: float = 1.0,
        w_clear: float = 0.5,
        clearance_cap: float = 0.5,
        precise_bias: float = 0.0,
        temperature: float = 1.0,
        z_clip: float = 3.0,
        d_safe: float = 0.3,
    ) -> None:
        self.w_eff = float(w_eff)
        self.w_clear = float(w_clear)
        self.clearance_cap = float(clearance_cap)
        self.precise_bias = float(precise_bias)
        self.temperature = float(temperature)
        self.z_clip = float(z_clip)
        self.d_safe = float(d_safe)

    # ------------------------------------------------------------------

    @staticmethod
    def _is_legal(branch) -> bool:
        """Precise is always legal; coarse only if the shield admitted it."""
        if branch.mode != COARSE:
            return True
        return branch.candidate is not None and bool(branch.candidate.safe)

    def _efficiency_scores(self, branches, legal: list[int]) -> np.ndarray:
        """MAD-standardised, softly saturated ``progress / step_cost`` per edge."""
        k = len(branches)
        z = np.zeros(k, dtype=np.float64)

        eff = np.full(k, np.nan, dtype=np.float64)
        for i in legal:
            progress = branches[i].progress
            if progress is None:
                continue                      # unknown → treated as the median
            eff[i] = float(progress) / max(float(branches[i].step_cost), 1e-9)

        known = eff[np.isfinite(eff)]
        if known.size == 0:
            return z

        # 1. Compress the multiplicative spread. arcsinh ~ log in the tail,
        #    ~ identity near 0, and defined for the negative eff of a group
        #    that would push its members away from their goals.
        eff_scale = float(np.median(np.abs(known)))
        if not eff_scale > 0.0:
            eff_scale = float(np.max(np.abs(known))) or 1.0
        u = np.arcsinh(eff / eff_scale)
        u_known = u[np.isfinite(u)]

        # 2. Standardise robustly, then saturate smoothly.
        median = float(np.median(u_known))
        mad = float(np.median(np.abs(u_known - median)))
        scale = _MAD_TO_SIGMA * mad + 1e-12
        cap = max(self.z_clip, 1e-9)
        for i in legal:
            if np.isfinite(u[i]):
                z[i] = cap * np.tanh((u[i] - median) / scale / cap)
        return z

    def _clearance_scores(
        self, branches, legal: list[int], d_safe: float
    ) -> np.ndarray:
        """Capped clearance margin, centred over the legal coarse edges."""
        k = len(branches)
        margin = np.zeros(k, dtype=np.float64)
        coarse = [i for i in legal if branches[i].mode == COARSE]
        if not coarse:
            return margin

        cap = max(self.clearance_cap, 1e-9)
        # Clamp before subtracting d_safe handles the +inf clearance a group
        # with nothing to hit reports.
        vals = np.array(
            [
                min(max(float(branches[i].candidate.clearance) - d_safe, 0.0), cap)
                / cap
                for i in coarse
            ],
            dtype=np.float64,
        )
        vals -= vals.mean()
        margin[coarse] = vals
        return margin

    # ------------------------------------------------------------------

    def __call__(self, model, ms, branches) -> np.ndarray:
        d_safe = float(getattr(model, "d_safe", self.d_safe))
        k = len(branches)
        logits = np.zeros(k, dtype=np.float64)

        legal = [i for i in range(k) if self._is_legal(branches[i])]
        if not legal:                          # cannot happen: precise is legal
            return logits

        z = self._efficiency_scores(branches, legal)
        margin = self._clearance_scores(branches, legal, d_safe)

        for i in legal:
            bias = self.precise_bias if branches[i].mode != COARSE else 0.0
            logits[i] = self.w_eff * z[i] + self.w_clear * margin[i] + bias
        return logits / max(self.temperature, 1e-9)


class UniformPrior:
    """Flat prior — every edge equally likely (ablation / tests / baseline)."""

    def __call__(self, model, ms, branches) -> np.ndarray:
        return np.zeros(len(branches), dtype=np.float64)
