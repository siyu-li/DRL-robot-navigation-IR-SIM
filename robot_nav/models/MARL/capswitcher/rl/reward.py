"""
Decision-level reward for the CAPSwitcher DQN training loop.

Unlike the previous substep-accumulation scheme, the switcher reward is computed
**once per switcher decision** (one ``SwitcherEnv.step``), not summed over the
sub-steps the chosen mode expands into.  This makes the reward agnostic to how
many sim sub-steps a mode consumes (coarse ≈ 10–14, precise = N×5 = 30 for the
sequential one-by-one plan), so the substep count no longer biases the value
function.

Reward shape
------------
    r =  k_p · Σ_i (d_start_i − d_end_i)        # progress, summed over robots
       − Σ_i cl_penalty_i                        # robot–robot proximity (decision end)
       − Σ_i obs_penalty_i                       # obstacle proximity   (decision end)
       + step_penalty(action)                    # coarse cheap, precise expensive
       + R_collision   if any collision          # terminal, large negative
       + R_allgoal     if all robots reached      # terminal, large positive

Progress is a telescoping quantity: over an episode with fixed goals it sums to
``initial_total_distance − final_total_distance``, identical for any trajectory
that reaches the goals.  So mode choice is driven by the step penalties and the
collision/clearance terms, which is exactly the physical-efficiency trade-off
the switcher should learn.

The collision and all-reached terminal events are exclusive of the shaping
terms (mirroring the phase-6 ``np.where`` override in the simulator): when one
fires, the reward is just the terminal value.
"""

from __future__ import annotations

# Action ids.
COARSE: int = 0
PRECISE: int = 1


class SwitcherReward:
    """
    Decision-level reward for the binary CAPSwitcher.

    Args:
        k_p:            Progress gain applied to the summed goal-distance
                        reduction over the decision (metres).
        coarse_penalty: Constant step penalty for a coarse decision (cheap).
        precise_penalty:Constant step penalty for a precise decision (expensive;
                        sequential one-by-one resolution wastes motion + time).
        r_collision:    Terminal reward when any robot collides.
        r_allgoal:      Terminal reward when all robots have reached their goals.
    """

    def __init__(
        self,
        k_p: float = 1.0,
        coarse_penalty: float = -0.5,
        precise_penalty: float = -3.0,
        r_collision: float = -100.0,
        r_allgoal: float = 200.0,
    ) -> None:
        self.k_p = k_p
        self.step_penalty = {COARSE: coarse_penalty, PRECISE: precise_penalty}
        self.r_collision = r_collision
        self.r_allgoal = r_allgoal

    def __call__(
        self,
        d_start,
        d_end,
        action: int,
        cl_pen,
        obs_pen,
        collision: bool,
        all_reached: bool,
    ) -> float:
        """
        Compute the scalar reward for one switcher decision.

        Args:
            d_start:     Per-robot goal distances at decision start (iterable).
            d_end:       Per-robot goal distances at decision end (iterable).
            action:      0 = coarse, 1 = precise.
            cl_pen:      Per-robot robot–robot proximity penalties at decision
                         end (iterable, already non-negative).
            obs_pen:     Per-robot obstacle proximity penalties at decision end
                         (iterable, already non-negative).
            collision:   True if any robot collided during the decision.
            all_reached: True if all robots reached their goals during the
                         decision.

        Returns:
            Scalar reward.
        """
        # Terminal events are exclusive of shaping (collision takes precedence).
        if collision:
            return float(self.r_collision)
        if all_reached:
            return float(self.r_allgoal)

        progress = float(sum(d_start) - sum(d_end))
        clearance = float(sum(cl_pen) + sum(obs_pen))
        return (
            self.k_p * progress
            - clearance
            + self.step_penalty[action]
        )


class StepPenaltyReward:
    """
    Terminal + step-penalty reward for the binary CAPSwitcher.

    A stripped-down alternative to :class:`SwitcherReward`: there is **no
    progress shaping and no proximity clearance term**.  Per non-terminal
    decision the reward is purely the constant step penalty for the chosen mode,
    so the switcher's mode preference is driven entirely by the (large) cost gap
    between precise and coarse, traded off against the terminal collision /
    all-reached events.

    Drop-in compatible with :class:`SwitcherReward`: identical ``__call__``
    signature, so it can be injected via ``SwitcherEnv(reward_fn=...)``.  The
    ``d_start``, ``d_end``, ``cl_pen`` and ``obs_pen`` arguments are accepted but
    ignored.

    Args:
        coarse_penalty:  Constant step penalty for a coarse decision (cheap).
        precise_penalty: Constant step penalty for a precise decision; make this
                         much larger in magnitude than ``coarse_penalty`` so the
                         switcher only pays for precise when it avoids a
                         collision.
        r_collision:     Terminal reward when any robot collides.
        r_allgoal:       Terminal reward when all robots have reached their goals.
    """

    def __init__(
        self,
        coarse_penalty: float = -0.5,
        precise_penalty: float = -5.0,
        r_collision: float = -100.0,
        r_allgoal: float = 200.0,
    ) -> None:
        self.step_penalty = {COARSE: coarse_penalty, PRECISE: precise_penalty}
        self.r_collision = r_collision
        self.r_allgoal = r_allgoal

    def __call__(
        self,
        d_start,
        d_end,
        action: int,
        cl_pen,
        obs_pen,
        collision: bool,
        all_reached: bool,
    ) -> float:
        """
        Compute the scalar reward for one switcher decision.

        Args (matching :class:`SwitcherReward`; shaping inputs are ignored):
            d_start, d_end, cl_pen, obs_pen: Accepted for signature
                         compatibility, unused.
            action:      0 = coarse, 1 = precise.
            collision:   True if any robot collided during the decision.
            all_reached: True if all robots reached their goals during the
                         decision.

        Returns:
            Scalar reward.
        """
        # Terminal events are exclusive of the step penalty (collision wins).
        if collision:
            return float(self.r_collision)
        if all_reached:
            return float(self.r_allgoal)

        return float(self.step_penalty[action])
