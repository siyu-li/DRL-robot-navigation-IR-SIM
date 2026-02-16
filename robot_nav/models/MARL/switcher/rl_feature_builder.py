"""
RL Feature Builder for PPO Group Switcher.

Separate feature builder for RL training (PPO) that produces:
  1. Per-group feature matrix X ∈ R^(M, D)  → Actor (policy head)
  2. State feature vector    s ∈ R^(S,)     → Critic (value head)

Differences from the supervised ``GroupFeatureBuilder``:
  - Fixed, curated 13-dim group scalars (not generic name→agg config).
  - ``coupling_mode`` is a first-class feature, not a flag.
  - Produces state-level features for the value head.
  - ``heading_error`` is a new per-robot feature (|heading - goal_direction|).

Per-group scalars (13):
  ┌─────────────────────────────────────────────────────────────┐
  │  Per-group (differentiate groups):                          │
  │  1. size_feat          — group_size / max_group_size        │
  │  2. coupling_mode      — 1.0 if rotation-coupled, else 0.0 │
  │  3. A_in               — intra-group attention              │
  │  4. A_out              — group-to-outside attention         │
  │  5. A_obs              — group-to-obstacle attention        │
  │  6. mean_dist_goal_g   — mean distance-to-goal of members  │
  │  7. min_dist_goal_g    — min distance-to-goal              │
  │  8. min_clearance_g    — worst clearance in group           │
  │  9. frac_reached_g     — fraction of group already reached  │
  │ 10. mean_heading_err_g — mean |heading - goal_direction|    │
  │                                                             │
  │ Global broadcast (context for credit assignment):           │
  │ 11. var_dist_goal      — global distance variance           │
  │ 12. frac_reached_global— global completion fraction         │
  │ 13. steps_elapsed_frac — time pressure signal               │
  └─────────────────────────────────────────────────────────────┘

State scalars for value head (5):
  [mean_dist, var_dist, min_clearance, frac_reached, steps_frac]
  All computed globally (across all robots, not per-group).
"""

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn


# Expected keys in the ``extra`` dict passed to ``forward()``/``build_state_features()``.
_REQUIRED_EXTRA_KEYS = [
    "dist_to_goal",         # (N,) per-robot distance to goal
    "clearance",            # (N,) per-robot min obstacle clearance
    "reached",              # (N,) per-robot binary reached flag
    "heading_error",        # (N,) per-robot |heading - goal_direction| in radians
    "frac_reached_global",  # (N,) broadcast: same scalar for every robot
    "var_dist_to_goal",     # (N,) broadcast
    "steps_elapsed_frac",   # (N,) broadcast
]

GROUP_SCALAR_DIM = 13
STATE_SCALAR_DIM = 5


class RLFeatureBuilder(nn.Module):
    """
    Feature builder for RL (PPO) group switcher training.

    Produces two outputs:
      - ``forward(...)`` → group features ``X ∈ R^(M, D)``
        where ``D = 2 * embed_dim + GROUP_SCALAR_DIM``.
      - ``build_state_features(...)`` → state features ``s ∈ R^(S,)``
        where ``S = embed_dim + STATE_SCALAR_DIM``.

    Args:
        embed_dim: Dimension of per-robot embeddings from the GAT backbone
            output (d).  The attention module produces ``H`` of shape
            ``(N, 2*embedding_dim)``; with ``embedding_dim=256`` the per-robot
            vector is 512-dim, so ``embed_dim`` should be **512**.
            ``h_g`` and ``h_glob`` each have this dim.
        global_embed_dim: Dimension of global embedding (dg).
            If *None*, defaults to ``embed_dim``.
        pooling: Pooling method for group embedding (``"mean"`` or ``"max"``).
        max_group_size: Normalisation constant for ``size_feat``.
            Default 7 (largest possible group in 6-robot m=3 binary allocation).
        rotation_coupling_threshold: Group sizes strictly above this get
            ``coupling_mode = 1.0``. Default 3.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        global_embed_dim: Optional[int] = None,
        pooling: Literal["mean", "max"] = "mean",
        max_group_size: int = 7,
        rotation_coupling_threshold: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_embed_dim = global_embed_dim if global_embed_dim is not None else embed_dim
        self.pooling = pooling
        self.max_group_size = max_group_size
        self.rotation_coupling_threshold = rotation_coupling_threshold

        # Pre-computed group index tensors — populated lazily on first
        # ``forward()`` call via ``_ensure_group_cache()``.  Avoids
        # rebuilding 60-group padded index / mask tensors every call.
        self._cached_groups = None
        self._gi_padded = None      # (M, max_gs) long
        self._mask = None           # (M, max_gs) float
        self._sizes = None          # (M,) float
        self._member_mask = None    # (M, N) float — for attention stats
        self._outside_mask = None   # (M, N) float
        self._in_mask_nodiag = None # (M, N, N) float
        self._out_mask = None       # (M, N, N) float
        self._mm_obs = None         # (M, N, 1) float

    # -----------------------------------------------------------------
    # Dimension properties
    # -----------------------------------------------------------------
    @property
    def group_scalar_dim(self) -> int:
        """Number of per-group scalar features (13)."""
        return GROUP_SCALAR_DIM

    @property
    def state_scalar_dim(self) -> int:
        """Number of state-level scalar features for the value head (5)."""
        return STATE_SCALAR_DIM

    @property
    def group_feature_dim(self) -> int:
        """Total dimension of per-group feature vector ``D = 2*embed_dim + 13``."""
        return 2 * self.embed_dim + GROUP_SCALAR_DIM

    @property
    def state_feature_dim(self) -> int:
        """Total dimension of state feature vector ``S = embed_dim + 5``."""
        return self.global_embed_dim + STATE_SCALAR_DIM

    # -----------------------------------------------------------------
    # Group features  →  Actor
    # -----------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        groups: List[List[int]],
        h_glob: Optional[torch.Tensor] = None,
        attn_rr: Optional[torch.Tensor] = None,
        attn_ro: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Build per-group feature matrix for the actor (policy head).

        Fully vectorized — all computation stays on GPU with no ``.item()``
        calls or Python-level per-element loops.

        Args:
            h: Per-robot embeddings ``(N, d)`` or ``(1, N, d)``.
            groups: List of M groups, each a list of robot indices.
            h_glob: Global embedding ``(dg,)`` or ``(1, dg)``.
                If *None*, computed as ``h.mean(dim=0)``.
            attn_rr: Robot-robot attention ``(N, N)`` or ``(1, N, N)``.
            attn_ro: Robot-obstacle attention ``(N, N_obs)`` or ``(1, N, N_obs)``.
            extra: Dict of per-robot features. Each value shape ``(N,)``.
                Required keys: see ``_REQUIRED_EXTRA_KEYS``.

        Returns:
            X: ``(M, D)`` group feature matrix.
        """
        h, h_glob, attn_rr, attn_ro, extra = self._unpack(
            h, h_glob, attn_rr, attn_ro, extra
        )
        device = h.device
        dtype = h.dtype
        n_robots = h.shape[0]
        M = len(groups)

        # --- Use pre-computed group index tensors (cached) ---
        self._ensure_group_cache(groups, n_robots, device, dtype)
        gi_padded = self._gi_padded    # (M, max_gs)
        mask = self._mask              # (M, max_gs)
        sizes = self._sizes            # (M,)

        # --- Per-group pooled embedding h_g: (M, d) ---
        h_gathered = h[gi_padded]  # (M, max_gs, d)
        if self.pooling == "mean":
            # Masked mean: sum(h * mask) / count
            h_g = (h_gathered * mask.unsqueeze(-1)).sum(dim=1) / sizes.unsqueeze(-1)
        else:
            # Masked max: fill masked positions with -inf, then max
            h_masked = h_gathered.clone()
            h_masked[mask == 0] = float("-inf")
            h_g = h_masked.max(dim=1)[0]

        # --- h_glob broadcast: (M, dg) ---
        h_glob_exp = h_glob.unsqueeze(0).expand(M, -1)

        # --- 13 scalar features (all on GPU, no .item()) ---
        # 1. size_feat
        size_feat = sizes / self.max_group_size  # (M,)

        # 2. coupling_mode
        coupling_mode = (sizes > self.rotation_coupling_threshold).to(dtype)  # (M,)

        # 3-5. attention stats [A_in, A_out, A_obs]
        attn_scalars = self._attention_stats_batched(
            groups, gi_padded, mask, sizes, n_robots, attn_rr, attn_ro, device, dtype,
        )  # (M, 3)

        # 6-10. extra per-group stats
        if extra is not None:
            dtg = extra["dist_to_goal"]   # (N,)
            clr = extra["clearance"]      # (N,)
            rch = extra["reached"]        # (N,)
            herr = extra["heading_error"] # (N,)

            dtg_g = dtg[gi_padded]   # (M, max_gs)
            clr_g = clr[gi_padded]
            rch_g = rch[gi_padded]
            herr_g = herr[gi_padded]

            # Masked mean / min
            big_val = 1e6
            dtg_masked = dtg_g * mask + (1 - mask) * big_val
            clr_masked = clr_g * mask + (1 - mask) * big_val
            rch_masked = rch_g * mask  # unreachable padded slots → 0
            herr_masked = herr_g * mask

            mean_dtg = (dtg_g * mask).sum(dim=1) / sizes          # (M,)
            min_dtg = dtg_masked.min(dim=1)[0]                     # (M,)
            min_clr = clr_masked.min(dim=1)[0]                     # (M,)
            frac_rch = (rch_masked).sum(dim=1) / sizes             # (M,)
            mean_herr = (herr_masked).sum(dim=1) / sizes           # (M,)

            extra_scalars = torch.stack(
                [mean_dtg, min_dtg, min_clr, frac_rch, mean_herr], dim=1
            )  # (M, 5)

            # 11-13. global broadcast scalars
            var_dtg = extra["var_dist_to_goal"][0].unsqueeze(0).expand(M)
            frac_rch_g = extra["frac_reached_global"][0].unsqueeze(0).expand(M)
            steps_f = extra["steps_elapsed_frac"][0].unsqueeze(0).expand(M)
            global_scalars = torch.stack([var_dtg, frac_rch_g, steps_f], dim=1)  # (M, 3)
        else:
            extra_scalars = torch.zeros(M, 5, device=device, dtype=dtype)
            global_scalars = torch.zeros(M, 3, device=device, dtype=dtype)

        # Stack all 13 scalars: (M, 13)
        all_scalars = torch.cat([
            size_feat.unsqueeze(1),       # (M, 1)
            coupling_mode.unsqueeze(1),   # (M, 1)
            attn_scalars,                 # (M, 3)
            extra_scalars,                # (M, 5)
            global_scalars,               # (M, 3)
        ], dim=1)

        return torch.cat([h_g, h_glob_exp, all_scalars], dim=1)  # (M, 2d + 13)

    # -----------------------------------------------------------------
    # State features  →  Critic (value head)
    # -----------------------------------------------------------------
    def build_state_features(
        self,
        h: torch.Tensor,
        h_glob: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Build state-level feature vector for the critic (value head).

        Fully vectorized — stays on GPU with no ``.item()`` round-trips.

        Layout: ``[h_glob (embed_dim) || state_scalars (5)]``

        State scalars (5):
          0. mean_dist_to_goal  — ``extra["dist_to_goal"].mean()``
          1. var_dist_to_goal   — ``extra["var_dist_to_goal"][0]`` (broadcast)
          2. min_clearance      — ``extra["clearance"].min()``
          3. frac_reached       — ``extra["frac_reached_global"][0]`` (broadcast)
          4. steps_elapsed_frac — ``extra["steps_elapsed_frac"][0]`` (broadcast)

        Args:
            h: Per-robot embeddings ``(N, d)`` or ``(1, N, d)``.
            h_glob: Global embedding ``(dg,)`` or ``(1, dg)``.
                If *None*, computed as ``h.mean(dim=0)``.
            extra: Dict of per-robot feature tensors.

        Returns:
            state: 1-D tensor ``(S,)``.
        """
        # Unpack batched dims
        if h.dim() == 3:
            h = h[0]
        if h_glob is None:
            h_glob = h.mean(dim=0)
        elif h_glob.dim() == 2:
            h_glob = h_glob[0]
        if extra is not None:
            extra = {k: (v[0] if v.dim() == 2 else v) for k, v in extra.items()}

        device = h.device
        dtype = h.dtype

        # 5 state scalars — all tensor ops, no .item()
        if extra is None:
            scalars = torch.zeros(STATE_SCALAR_DIM, device=device, dtype=dtype)
        else:
            scalars = torch.stack([
                extra["dist_to_goal"].mean(),
                extra["var_dist_to_goal"][0],
                extra["clearance"].min(),
                extra["frac_reached_global"][0],
                extra["steps_elapsed_frac"][0],
            ])

        return torch.cat([h_glob, scalars], dim=0)  # (dg + 5,)

    # =================================================================
    # Internals
    # =================================================================
    def _ensure_group_cache(
        self, groups: List[List[int]], n_robots: int, device: torch.device, dtype: torch.dtype,
    ):
        """Build (or reuse) pre-computed group index tensors.

        These are constant for a given set of groups and only need to be
        built once.  Rebuilding is triggered if ``groups`` identity changes.
        """
        if self._cached_groups is groups and self._gi_padded is not None:
            # Already cached for this exact groups list — check device
            if self._gi_padded.device == device:
                return
        # Build fresh cache
        M = len(groups)
        max_gs = max(len(g) for g in groups)

        gi_padded = torch.zeros(M, max_gs, device=device, dtype=torch.long)
        mask = torch.zeros(M, max_gs, device=device, dtype=dtype)
        sizes = torch.zeros(M, device=device, dtype=dtype)
        member_mask = torch.zeros(M, n_robots, device=device, dtype=dtype)

        for m, group in enumerate(groups):
            gs = len(group)
            gi = torch.tensor(group, device=device, dtype=torch.long)
            gi_padded[m, :gs] = gi
            mask[m, :gs] = 1.0
            sizes[m] = gs
            member_mask[m, gi] = 1.0

        outside_mask = 1.0 - member_mask  # (M, N)

        # Attention stat masks — pre-compute once
        mm_row = member_mask.unsqueeze(2)   # (M, N, 1)
        mm_col = member_mask.unsqueeze(1)   # (M, 1, N)
        in_mask = mm_row * mm_col           # (M, N, N)
        eye = torch.eye(n_robots, device=device, dtype=dtype).unsqueeze(0)
        in_mask_nodiag = in_mask * (1.0 - eye)
        out_mask = mm_row * outside_mask.unsqueeze(1)  # (M, N, N)
        mm_obs = member_mask.unsqueeze(2)  # (M, N, 1)

        self._cached_groups = groups
        self._gi_padded = gi_padded
        self._mask = mask
        self._sizes = sizes
        self._member_mask = member_mask
        self._outside_mask = outside_mask
        self._in_mask_nodiag = in_mask_nodiag
        self._out_mask = out_mask
        self._mm_obs = mm_obs

    def _unpack(self, h, h_glob, attn_rr, attn_ro, extra):
        """Squeeze batch dims and fill defaults."""
        if h.dim() == 3:
            h = h[0]
        if h_glob is None:
            h_glob = h.mean(dim=0)
        elif h_glob.dim() == 2:
            h_glob = h_glob[0]
        if attn_rr is not None and attn_rr.dim() == 3:
            attn_rr = attn_rr[0]
        if attn_ro is not None and attn_ro.dim() == 3:
            attn_ro = attn_ro[0]
        if extra is not None:
            extra = {k: (v[0] if v.dim() == 2 else v) for k, v in extra.items()}
        return h, h_glob, attn_rr, attn_ro, extra

    def _build_group_row(
        self,
        h: torch.Tensor,
        h_glob: torch.Tensor,
        group: List[int],
        n_robots: int,
        attn_rr: Optional[torch.Tensor],
        attn_ro: Optional[torch.Tensor],
        extra: Optional[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """Build one row of the group feature matrix ``[h_g, h_glob, scalars_13]``.

        NOTE: This is the legacy per-group path kept for compatibility.
        The vectorized ``forward()`` above is used in the hot path.
        """
        dtype = h.dtype
        gi = torch.tensor(group, device=device, dtype=torch.long)
        gs = len(group)

        # ---- Embedding features ----
        h_group = h[gi]  # (|g|, d)
        if self.pooling == "mean":
            h_g = h_group.mean(dim=0)
        else:
            h_g = h_group.max(dim=0)[0]

        # ---- 13 scalar features (vectorized, no .item()) ----
        scalars_list: List[torch.Tensor] = []

        # 1. size_feat
        scalars_list.append(torch.tensor([gs / self.max_group_size], device=device, dtype=dtype))

        # 2. coupling_mode
        scalars_list.append(torch.tensor([1.0 if gs > self.rotation_coupling_threshold else 0.0], device=device, dtype=dtype))

        # 3-5. attention stats [A_in, A_out, A_obs]
        a_in, a_out, a_obs = self._attention_stats_vectorized(group, n_robots, attn_rr, attn_ro, device, dtype)
        scalars_list.append(torch.stack([a_in, a_out, a_obs]))

        # 6-10. extra per-group stats
        if extra is not None:
            dtg_g = extra["dist_to_goal"][gi]
            scalars_list.append(torch.stack([
                dtg_g.mean(),
                dtg_g.min(),
                extra["clearance"][gi].min(),
                extra["reached"][gi].mean(),
                extra["heading_error"][gi].mean(),
            ]))
        else:
            scalars_list.append(torch.zeros(5, device=device, dtype=dtype))

        # 11-13. global broadcast scalars
        if extra is not None:
            scalars_list.append(torch.stack([
                extra["var_dist_to_goal"][0],
                extra["frac_reached_global"][0],
                extra["steps_elapsed_frac"][0],
            ]))
        else:
            scalars_list.append(torch.zeros(3, device=device, dtype=dtype))

        scalar_t = torch.cat(scalars_list)  # (13,)

        return torch.cat([h_g, h_glob, scalar_t], dim=0)  # (2d + 13,)

    def _attention_stats_batched(
        self,
        groups: List[List[int]],
        gi_padded: torch.Tensor,
        mask: torch.Tensor,
        sizes: torch.Tensor,
        n_robots: int,
        attn_rr: Optional[torch.Tensor],
        attn_ro: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute [A_in, A_out, A_obs] for all groups in a batched manner.

        Uses pre-computed masks from ``_ensure_group_cache()``.

        Returns (M, 3) tensor on ``device``.
        """
        M = len(groups)
        if attn_rr is None:
            return torch.zeros(M, 3, device=device, dtype=dtype)

        # Use pre-computed masks (no Python loops)
        in_mask_nodiag = self._in_mask_nodiag   # (M, N, N)
        out_mask = self._out_mask               # (M, N, N)

        attn_rr_expanded = attn_rr.unsqueeze(0)  # (1, N, N)

        # A_in: mean intra-group attention (excluding self)
        a_in_sum = (attn_rr_expanded * in_mask_nodiag).sum(dim=(1, 2))  # (M,)
        in_count = sizes * (sizes - 1)
        a_in = torch.where(in_count > 0, a_in_sum / in_count, torch.zeros_like(a_in_sum))

        # A_out: mean attention from group members to outside robots
        a_out_sum = (attn_rr_expanded * out_mask).sum(dim=(1, 2))
        n_outside = (n_robots - sizes)
        out_count = sizes * n_outside
        a_out = torch.where(out_count > 0, a_out_sum / out_count, torch.zeros_like(a_out_sum))

        # A_obs
        if attn_ro is not None:
            n_obs = attn_ro.shape[1]
            attn_ro_expanded = attn_ro.unsqueeze(0)  # (1, N, N_obs)
            mm_obs = self._mm_obs                     # (M, N, 1)
            a_obs_sum = (attn_ro_expanded * mm_obs).sum(dim=(1, 2))  # (M,)
            obs_count = sizes * n_obs
            a_obs = torch.where(obs_count > 0, a_obs_sum / obs_count, torch.zeros_like(a_obs_sum))
        else:
            a_obs = torch.zeros(M, device=device, dtype=dtype)

        return torch.stack([a_in, a_out, a_obs], dim=1)  # (M, 3)

    @staticmethod
    def _attention_stats_vectorized(
        group: List[int],
        n_robots: int,
        attn_rr: Optional[torch.Tensor],
        attn_ro: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        """Compute [A_in, A_out, A_obs] for a single group using tensor ops."""
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        if attn_rr is None:
            return (zero, zero, zero)

        gi = torch.tensor(group, device=device, dtype=torch.long)
        gs = len(group)

        # Build boolean masks
        member = torch.zeros(n_robots, device=device, dtype=torch.bool)
        member[gi] = True
        outside = ~member

        # A_in: attn between group members excluding diagonal
        if gs > 1:
            block = attn_rr[gi][:, gi]  # (gs, gs)
            # Subtract diagonal
            a_in = (block.sum() - block.diag().sum()) / (gs * (gs - 1))
        else:
            a_in = zero

        # A_out: attn from group to outside
        n_outside = int(outside.sum().item())
        if n_outside > 0:
            outside_idx = outside.nonzero(as_tuple=True)[0]
            a_out = attn_rr[gi][:, outside_idx].mean()
        else:
            a_out = zero

        # A_obs
        if attn_ro is not None:
            a_obs = attn_ro[gi].mean()
        else:
            a_obs = zero

        return (a_in, a_out, a_obs)

    @staticmethod
    def _attention_stats(
        group: List[int],
        n_robots: int,
        attn_rr: Optional[torch.Tensor],
        attn_ro: Optional[torch.Tensor],
    ) -> tuple:
        """Compute [A_in, A_out, A_obs] for a single group."""
        if attn_rr is None:
            return (0.0, 0.0, 0.0)

        group_set = set(group)
        outside = [j for j in range(n_robots) if j not in group_set]

        # A_in
        if len(group) > 1:
            a_in_sum = sum(
                attn_rr[i, j].item() for i in group for j in group if i != j
            )
            a_in = a_in_sum / (len(group) * (len(group) - 1))
        else:
            a_in = 0.0

        # A_out
        if outside:
            a_out_sum = sum(
                attn_rr[i, j].item() for i in group for j in outside
            )
            a_out = a_out_sum / (len(group) * len(outside))
        else:
            a_out = 0.0

        # A_obs
        if attn_ro is not None:
            n_obs = attn_ro.shape[1]
            a_obs_sum = sum(attn_ro[i, o].item() for i in group for o in range(n_obs))
            a_obs = a_obs_sum / max(len(group) * n_obs, 1)
        else:
            a_obs = 0.0

        return (a_in, a_out, a_obs)
