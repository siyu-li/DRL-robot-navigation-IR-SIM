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
        coarse_cost: float = 0.0,
        oob: bool = False,
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
            coarse_cost: Accepted for signature compatibility with
                         :class:`PathCostReward`; unused.
            oob:         Accepted for signature compatibility; unused.

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
        coarse_cost: float = 0.0,
        oob: bool = False,
    ) -> float:
        """
        Compute the scalar reward for one switcher decision.

        Args (matching :class:`SwitcherReward`; shaping inputs are ignored):
            d_start, d_end, cl_pen, obs_pen, coarse_cost, oob: Accepted for
                         signature compatibility, unused.
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


class PathCostReward:
    """
    Executed-path-cost reward for the binary CAPSwitcher.

    The switcher's objective is reframed as **minimising the total executed
    motion cost** required to drive every robot to its goal.  There is no
    progress shaping and no proximity-clearance term: each non-terminal
    decision is penalised by exactly the motion cost the chosen mode actually
    executed, and the terminal goal / collision events supply the large
    signals that make reaching the goal worthwhile and collisions catastrophic.

    Per-mode executed cost
    ----------------------
    * **Coarse** — the hard-coded :class:`CoarseSteering` advances the chosen
      group's members forward by a fixed ``move_distance`` each, while
      non-members only rotate in place.  The executed path length is therefore

          coarse_cost = (number of members that moved) · move_distance

      This is supplied per decision by the environment (it knows the group and
      ``move_distance``) and passed in as ``coarse_cost``.

    * **Precise** — under the one-robot-at-a-time precise mechanism every
      precise decision corresponds to a fixed executed motion cost,
      independent of the apparent state-space travel distance.  That constant
      is :attr:`precise_cost` (default 11.7).

    Reward shape
    ------------
        r = −cost_scale · executed_cost            # non-terminal decision
        r = r_allgoal       if all robots reached  # terminal, large positive
        r = r_collision     if any collision       # terminal, large negative
        r = r_invalid       if out-of-bounds       # terminal, large negative

    Terminal events are exclusive of the motion-cost term (collision takes
    precedence over out-of-bounds, which takes precedence over all-reached).

    Drop-in compatible with :class:`SwitcherReward` / :class:`StepPenaltyReward`
    (same ``__call__`` signature plus the ``coarse_cost`` / ``oob`` keywords the
    others ignore), so it can be injected via ``SwitcherEnv(reward_fn=...)``.

    Args:
        precise_cost: Fixed executed motion cost of a precise decision
                      (default 11.7, per the project's modelling choice).
        cost_scale:   Multiplier on the executed cost before negating (default
                      1.0).  Scales the cost term relative to the terminal
                      rewards.
        r_collision:  Terminal reward when any robot collides.
        r_allgoal:    Terminal reward when all robots have reached their goals.
        r_invalid:    Terminal reward for an invalid motion (e.g. a robot
                      leaving the world bounds).  Defaults to ``r_collision``.
    """

    def __init__(
        self,
        precise_cost: float = 11.7,
        cost_scale: float = 1.0,
        r_collision: float = -100.0,
        r_allgoal: float = 200.0,
        r_invalid: float | None = None,
    ) -> None:
        self.precise_cost = precise_cost
        self.cost_scale = cost_scale
        self.r_collision = r_collision
        self.r_allgoal = r_allgoal
        self.r_invalid = r_collision if r_invalid is None else r_invalid

    def __call__(
        self,
        d_start,
        d_end,
        action: int,
        cl_pen,
        obs_pen,
        collision: bool,
        all_reached: bool,
        coarse_cost: float = 0.0,
        oob: bool = False,
    ) -> float:
        """
        Compute the scalar reward for one switcher decision.

        Args:
            d_start, d_end, cl_pen, obs_pen: Accepted for signature
                         compatibility with :class:`SwitcherReward`; unused
                         (no progress or clearance shaping).
            action:      0 = coarse, 1 = precise.
            collision:   True if any robot collided during the decision.
            all_reached: True if all robots reached their goals during the
                         decision.
            coarse_cost: Executed path length of a coarse decision
                         (= n_members_moved · move_distance).  Ignored for
                         precise decisions, which use the fixed
                         :attr:`precise_cost`.
            oob:         True if a robot left the world bounds during the
                         decision (treated as an invalid motion).

        Returns:
            Scalar reward.
        """
        # Terminal events are exclusive of the motion-cost term.
        if collision:
            return float(self.r_collision)
        if oob:
            return float(self.r_invalid)
        if all_reached:
            return float(self.r_allgoal)

        cost = float(coarse_cost) if action == COARSE else self.precise_cost
        return -self.cost_scale * cost
