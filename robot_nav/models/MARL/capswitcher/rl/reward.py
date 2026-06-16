"""
Reward constants for the CAPSwitcher PPO training loop.

Mode costs encode the physical efficiency trade-off between the two
control regimes:

  Coarse control  — all robots advance simultaneously; small per-step cost.
  Precise control — back-and-forth reversal wastes physical motion; large cost.

The switcher is rewarded by the standard simulation reward (progress toward
goal, collision penalties, goal bonus) *plus* the mode cost at each sub-step.
This makes precise control more expensive, so the learner prefers coarse when
it is safe to do so.
"""

# Per sim-step cost added to the simulation reward.
# action 0 → coarse (cheap),  action 1 → precise (expensive).
COARSE_COST: float = -0.2
PRECISE_COST: float = -1.0

MODE_COSTS: dict[int, float] = {
    0: COARSE_COST,
    1: PRECISE_COST,
}
