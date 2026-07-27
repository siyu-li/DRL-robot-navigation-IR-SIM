# CAPSwitcher — system and problem formulation

Coarse-And-Precise Switcher for coupled unicycle swarms. This file holds the **stable** definition of the system and the problem. 

Stated for `N` robots and `K` groups. The current experimental instantiation is `N=6, K=3`
(§5); scaling `N` is a target of the work, so nothing here should assume `N=6`.

---

## 1. The physical system

A swarm of `N` coupled unicycle robots organized into overlapping groups. Each group defines a
collective actuation mode. Activating a group makes **all robots in the swarm rotate** — not
just the group's members. Rotation per robot is set by the actuation matrix `A`.

**Membership matrix** `M ∈ {0,1}^{K×N}`, where `M[s,i] = 1` means robot `i` belongs to group `s`.
Example for `N=6`, `K=4`:

| | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|
| Group 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| Group 2 | 0 | 1 | 1 | 0 | 0 | 1 |
| Group 3 | 1 | 0 | 1 | 0 | 1 | 0 |
| Group 4 | 0 | 0 | 0 | 0 | 0 | 0 |

**Actuation matrix** `A` is the entrywise **complement** of `Mᵀ` (1 → 0, 0 → 1): the robots that
do *not* belong to a group are the ones that rotate when it is activated.

| | G1 | G2 | G3 | G4 |
|---|---|---|---|---|
| R1 | **1** | **1** | 0 | **1** |
| R2 | **1** | 0 | **1** | **1** |
| R3 | **1** | 0 | 0 | **1** |
| R4 | 0 | **1** | **1** | **1** |
| R5 | 0 | **1** | 0 | **1** |
| R6 | 0 | 0 | **1** | **1** |

An all-zero membership row (G4 above) complements to an all-ones column in `A` — a mode where
every robot rotates by the same angle.

**Dynamics of one group activation.** Activating group `s` with rotation parameter `t` rotates
each robot by `dθ_i = A[i,s]·t`; then the group's **members** advance along their individual
headings by a fixed distance `d` while non-members hold still. Activating several groups with
parameters `t = [t₁…t_K]` gives total rotation `dθ = A·t`.

## 2. The rank-deficiency problem

`A` is rank-deficient — fewer independent columns than robots — so the system **cannot
independently control all robot headings**. Only a low-dimensional subspace of the full rotation
space is reachable by any combination of group activations. This is the core physical
limitation the switcher exists to work around.

## 3. Two control regimes

**Coarse group control** — [`policies/coarse_steering.py`](../policies/coarse_steering.py).
Two phases: (1) *rotation*, all robots rotate per `A`; (2) *move*, only the chosen group's
members advance by `move_distance`, non-members hold still. Physically efficient — moving
robots make forward progress with no reversal. But rank deficiency means headings can only be
adjusted inside the controllable subspace, so **independent steering and obstacle avoidance are
not possible at this level**.

Given desired goal-facing headings, the group rotation solves one of:

```
least_squares:  t* = pinv(A_reduced) · dθ_desired                    # closest reachable headings
nonlinear:      t* = argmax Σ_{i∈members} Δdistance_to_goal_i(t)     # location progress, BFGS
```

then applies `dθ_actual = A_reduced · t*` to **all** robots. The nonlinear variant is
initialised at the least-squares solution; `last_solve_time_s` records per-call solve time for
the LS-vs-nonlinear runtime comparison.

*Implementation notes.* Each phase is split into velocity-bounded **sub-step frames**, so a
large rotation/translation is realised over several simulator steps within the angular/linear
velocity caps. `A_reduced` is rebuilt per call — the chosen group's own column is dropped, then
one of the remaining columns is dropped — which is how the artificially rank-2 system of §5 is
produced.

**Precise (dense) control** — [`policies/gat_backbone.py`](../policies/gat_backbone.py).
Conceptually it composes multiple group activations (activate a coarse group, then further
groups that rotate unwanted robots by 180° and move them backward to cancel their motion) —
physically wasteful because of the back-and-forth reversal. In implementation it is the frozen
GAT policy run **sequentially, one robot at a time**: each robot is driven by its individual
obstacle-aware GAT action for `selection_interval` sub-steps while all others hold still.
Resolving robots one-by-one is the source of the inefficiency the switcher weighs against
coarse control.

**The trade-off.** Coarse = high physical efficiency, low precision (no obstacle avoidance).
Precise = high precision, low physical efficiency. Always-precise reaches the goal but wastes
motion; always-coarse is efficient but collides and cannot navigate precisely.

## 4. The problem

Decide, at each decision point, whether to use coarse group control or precise individual
control, so as to **minimize precise usage / physical path cost subject to safety**. The
intended behaviour: coarse in open space where distance must be covered efficiently, precise
near obstacles and near the goal where individual steering matters.

Primary goal: demonstrate that a switcher choosing between coarse steering and the GAT-based
precise policy produces more physically efficient multi-robot navigation than either mode
alone — and that the mechanism scales in `N`.

## 5. Current instantiation

`N=6` robots, `K=3` coarse groups, artificially constrained to **rank 2** (only 2 of 3 groups
used, via the reduced-matrix construction in §3) in order to study the rank-deficiency
limitation in a controlled setting.

---

## 6. GAT backbone analysis
The precise controller and the switcher's observation both come from `AttentionObstacleOptimized`
([`Attention/iga_obstacle_optimized.py`](../../Attention/iga_obstacle_optimized.py)), used inside
`ActorObstacle` ([`marlTD3/marlTD3_obstacle.py`](../../marlTD3/marlTD3_obstacle.py)).
**Frozen everywhere it is used.**

