"""
Learned Action Coupling via Softmax Mixing Weights.

Replaces the mean-based coupling in ``action_coupling.py`` with a learned
softmax-weighted shared velocity computed by ``MixingNetwork``.

Protocol
--------
Every *group_interval* steps an active group **G** is chosen (uniformly at
random from all generated groups).  For that group:

1. Run the **frozen** GAT + TD3 actor to get per-robot embeddings ``e_i``
   and raw actions ``(v_i, ω_i)``.
2. ``w_i = f_lin(e_i)`` for each robot *i* in *G*.
3. Hard-mask arrived robots to ``-∞``.
4. Softmax (optionally scaled by ``√|G|``) over the group to get ``α_i``.
5. ``v_shared = Σ α_i · v_i`` — applied to every robot in the group.
6. For groups with ``|G| ≤ rotation_coupling_threshold``, each robot keeps
   its own ``ω_i``.  For larger groups, ``ω_shared = Σ α_i · ω_i``.

Robots **not** in the active group receive ``[0, 0]`` actions.

Two entry points
-----------------
- ``compute_mixed_actions``:        returns ``List[List[float]]`` for env stepping (no grad).
- ``compute_mixed_actions_tensor``: returns ``Tensor (N, 2)`` that keeps the graph alive
                                    through ``f_lin`` so the frozen critic's ``-Q`` gradient
                                    can flow into the mixing network.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from robot_nav.models.MARL.groups.mixing_network import MixingNetwork


# ---------------------------------------------------------------------------
# Internal helper: shared softmax logic
# ---------------------------------------------------------------------------

def _compute_alpha(
    mixing_net: MixingNetwork,
    group_embeds: torch.Tensor,
    group: List[int],
    arrived_mask: Optional[List[bool]],
    scale_by_sqrt: bool,
) -> torch.Tensor:
    """Return softmax mixing weights α of shape ``(G,)`` with grad."""
    device = group_embeds.device
    group_size = len(group)

    logits = mixing_net(group_embeds)  # (G,)

    # Mask arrived robots
    if arrived_mask is not None:
        for local_idx, robot_idx in enumerate(group):
            if arrived_mask[robot_idx]:
                logits[local_idx] = float("-inf")

    # Safety: if all masked, fall back to uniform
    if torch.all(logits == float("-inf")):
        logits = torch.zeros(group_size, device=device)

    # √|G| scaling
    if scale_by_sqrt and group_size > 1:
        logits = logits * math.sqrt(group_size)

    return F.softmax(logits, dim=0)


# ---------------------------------------------------------------------------
# 1. For env stepping — returns plain python lists (no grad needed)
# ---------------------------------------------------------------------------

def compute_mixed_actions(
    mixing_net: MixingNetwork,
    embeddings: torch.Tensor,
    raw_actions: np.ndarray,
    group: List[int],
    num_robots: int,
    arrived_mask: Optional[List[bool]] = None,
    rotation_coupling_threshold: int = 3,
    scale_by_sqrt: bool = True,
) -> List[List[float]]:
    """
    Produce coupled actions for a group using the learned mixing network.

    Used during **data collection** (env stepping).  Returns plain python
    lists — no gradient graph is retained.

    Args:
        mixing_net: The ``MixingNetwork`` that maps embeddings to logits.
        embeddings: All per-robot embeddings from the frozen actor,
            shape ``(num_robots, embed_dim)``.
        raw_actions: Pre-computed raw policy output, shape ``(num_robots, 2)``,
            values in ``[-1, 1]``.
        group: Indices of robots in the active group.
        num_robots: Total number of robots.
        arrived_mask: Boolean list of length ``num_robots``.
        rotation_coupling_threshold: Groups with ``|G| > threshold`` also
            share angular velocity.  Default 3.
        scale_by_sqrt: Scale logits by ``√|G|``.

    Returns:
        ``List[List[float]]`` of length ``num_robots``.
    """
    device = next(mixing_net.parameters()).device
    group_size = len(group)
    group_set = set(group)

    if group_size == 0:
        return [[0.0, 0.0]] * num_robots

    if group_size == 1:
        idx = group[0]
        v = (raw_actions[idx][0] + 1) / 4
        a_out = [[0.0, 0.0]] * num_robots
        a_out[idx] = [v, float(raw_actions[idx][1])]
        return a_out

    # Gather group embeddings
    g_idx = torch.tensor(group, dtype=torch.long, device=device)
    g_emb = embeddings[g_idx]

    with torch.no_grad():
        alpha = _compute_alpha(mixing_net, g_emb, group, arrived_mask, scale_by_sqrt)

        scaled_vels = torch.tensor(
            [(raw_actions[idx][0] + 1) / 4 for idx in group],
            dtype=torch.float32, device=device,
        )
        v_shared = float((alpha * scaled_vels).sum())

        if group_size > rotation_coupling_threshold:
            ang_vels = torch.tensor(
                [raw_actions[idx][1] for idx in group],
                dtype=torch.float32, device=device,
            )
            w_shared: Optional[float] = float((alpha * ang_vels).sum())
        else:
            w_shared = None

    a_out: List[List[float]] = []
    for i in range(num_robots):
        if i in group_set:
            w = w_shared if w_shared is not None else float(raw_actions[i][1])
            a_out.append([v_shared, w])
        else:
            a_out.append([0.0, 0.0])
    return a_out


# ---------------------------------------------------------------------------
# 2. For training — returns (N, 2) Tensor with grad through f_lin
# ---------------------------------------------------------------------------

def compute_mixed_actions_tensor(
    mixing_net: MixingNetwork,
    embeddings: torch.Tensor,
    raw_actions_tensor: torch.Tensor,
    group: List[int],
    num_robots: int,
    arrived_mask: Optional[List[bool]] = None,
    rotation_coupling_threshold: int = 3,
    scale_by_sqrt: bool = True,
) -> torch.Tensor:
    """
    Differentiable version of ``compute_mixed_actions``.

    Returns a ``(num_robots, 2)`` tensor where gradient flows through the
    mixing weights ``α`` into ``f_lin`` parameters.  Used during the TD3-style
    training step so that ``loss = -Q_frozen(s, a_coupled)`` can backprop
    into the mixing network.

    For robots **not** in the group, the original (frozen) per-robot action
    is kept unchanged (detached, no grad contribution).

    Args:
        mixing_net: The ``MixingNetwork``.
        embeddings: Per-robot embeddings, ``(num_robots, embed_dim)``.
            Detached from the frozen actor but will flow into ``mixing_net``.
        raw_actions_tensor: Frozen actor raw actions, ``(num_robots, 2)``
            **as a Tensor** (detached).  Values in ``[-1, 1]``.
        group: Robot indices in the active group.
        num_robots: Total robot count.
        arrived_mask: Boolean list; ``True`` ⟹ mask to ``-∞``.
        rotation_coupling_threshold: ``|G| > threshold`` ⟹ share ω too.
        scale_by_sqrt: Scale logits by ``√|G|``.

    Returns:
        Tensor of shape ``(num_robots, 2)`` with grad only through ``f_lin``.
    """
    device = embeddings.device
    group_size = len(group)

    # Start from the frozen per-robot actions (no grad into actor)
    actions_out = raw_actions_tensor.detach().clone()  # (N, 2)

    if group_size <= 1:
        return actions_out  # nothing to mix

    g_idx = torch.tensor(group, dtype=torch.long, device=device)
    g_emb = embeddings[g_idx]  # (G, embed_dim) — detached from actor

    # α has grad through f_lin
    alpha = _compute_alpha(mixing_net, g_emb, group, arrived_mask, scale_by_sqrt)

    # Group raw actions (detached)
    g_actions = raw_actions_tensor[g_idx].detach()  # (G, 2)

    # Shared linear velocity:  v_shared = Σ α_i · v_i
    v_shared = (alpha * g_actions[:, 0]).sum()  # scalar, has grad

    # Assign v_shared to all group members
    for local_idx, robot_idx in enumerate(group):
        actions_out[robot_idx, 0] = v_shared

    # Shared angular velocity for large groups
    if group_size > rotation_coupling_threshold:
        w_shared = (alpha * g_actions[:, 1]).sum()
        for robot_idx in group:
            actions_out[robot_idx, 1] = w_shared

    return actions_out  # (N, 2), grad flows through v_shared → α → f_lin


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_embeddings_from_frozen_actor(
    actor: torch.nn.Module,
    robot_obs: np.ndarray,
    obstacle_obs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the frozen actor's attention + decode layers and return the per-robot
    embeddings (before the policy head).

    The returned tensor has shape ``(num_robots, embedding_dim * 2)`` and is
    **detached** (no gradients flow into the actor).

    Args:
        actor: The frozen ``ActorObstacle`` module.
        robot_obs: Robot observations, shape ``(num_robots, state_dim)``.
        obstacle_obs: Obstacle observations, shape ``(num_obstacles, obs_dim)``.
        device: Torch device.

    Returns:
        Tensor: Per-robot embeddings, shape ``(num_robots, embedding_dim * 2)``.
    """
    robot_state = torch.Tensor(robot_obs).to(device)
    obstacle_state = torch.Tensor(obstacle_obs).to(device)

    with torch.no_grad():
        (
            attn_out,
            _, _, _, _, _, _, _, _,
        ) = actor.attention(robot_state, obstacle_state)
        embeddings = attn_out.detach()

    return embeddings
