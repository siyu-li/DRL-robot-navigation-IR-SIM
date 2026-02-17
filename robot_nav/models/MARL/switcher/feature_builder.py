"""
Group Feature Builder for Ranking-Based Group Switcher (Supervised Learning).

Constructs a configurable scalar feature vector for each candidate group.
Features 1–5 are always included; features 6–13 are optional and controlled
by two config lists:

  **Always-on scalars** (features 1–5, per-group):
    1. size_feat          — group_size / max_group_size
    2. coupling_mode      — 1.0 if rotation-coupled, else 0.0
    3. A_in               — intra-group attention
    4. A_out              — group-to-outside attention
    5. A_obs              — group-to-obstacle attention

  **extra_group** — per-group extras (aggregated over group members):
    Each entry is ``(key, agg)`` where *key* is a per-robot ``extra`` tensor
    and *agg* ∈ {"mean", "min", "max", "sum"}.  Defaults to:
      ("dist_to_goal", "mean")   → mean_dist_goal_g
      ("dist_to_goal", "min")    → min_dist_goal_g
      ("clearance",    "min")    → min_clearance_g
      ("reached",      "mean")   → frac_reached_g
      ("heading_error","mean")   → mean_heading_err_g

  **extra_global** — global context (same value for every group):
    Each entry is a key name whose ``extra`` tensor has shape ``(1,)``.
    Defaults to:
      "var_dist_goal_global"     → distance variance across all robots
      "frac_reached_global"      → global completion fraction
      "steps_elapsed_frac"       → time pressure signal

  scalar_dim = 5 + len(extra_group) + len(extra_global)

The ``extra`` dict passed to ``forward()`` must contain the per-robot
``(N,)`` tensors and global ``(1,)`` scalars referenced by
``extra_group`` and ``extra_global``.
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

# ── Constants ──
_BASE_SCALAR_DIM = 5  # size_feat, coupling_mode, A_in, A_out, A_obs

# Default extra_group: list of (extra_key, aggregation)
DEFAULT_EXTRA_GROUP: List[Tuple[str, str]] = [
    ("dist_to_goal", "mean"),   # mean_dist_goal_g
    ("dist_to_goal", "min"),    # min_dist_goal_g
    ("clearance",    "min"),    # min_clearance_g
    ("reached",      "mean"),   # frac_reached_g
    ("heading_error","mean"),   # mean_heading_err_g
]

# Default extra_global: list of extra keys (each stored as shape (1,))
DEFAULT_EXTRA_GLOBAL: List[str] = [
    "var_dist_goal_global",     # distance variance (sync signal)
    "frac_reached_global",      # global completion fraction
    "steps_elapsed_frac",       # time pressure
]

_VALID_AGGS = {"mean", "min", "max", "sum"}


class GroupFeatureBuilder(nn.Module):
    """
    Builds feature vectors for candidate groups from per-robot embeddings,
    attention weights, per-robot extra features, and global scalars.

    Output feature format per group (D-dim vector):
        [h_g, h_glob, scalars]

    Where:
        - h_g:    Pooled group embedding        (embed_dim)
        - h_glob: Global embedding              (global_embed_dim, defaults to embed_dim)
        - scalars: 5 base + len(extra_group) + len(extra_global)

    Args:
        embed_dim: Dimension of per-robot embeddings from GAT backbone output.
        global_embed_dim: Dimension of global embedding. If None, uses embed_dim.
        pooling: Pooling method for group embedding ("mean" or "max").
        max_group_size: Normalisation constant for size_feat. Default 7.
        rotation_coupling_threshold: Group sizes strictly above this get
            coupling_mode = 1.0. Default 3 (sizes 4, 7 get coupling_mode=1).
        extra_group: Per-group extra features, each ``(key, agg)``.
            ``key`` references a per-robot tensor ``(N,)`` in ``extra``; ``agg``
            is one of ``"mean", "min", "max", "sum"``.
            Pass ``[]`` to disable.  ``None`` → DEFAULT_EXTRA_GROUP.
        extra_global: Global extra features, each a key referencing a ``(1,)``
            tensor in ``extra``.
            Pass ``[]`` to disable.  ``None`` → DEFAULT_EXTRA_GLOBAL.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        global_embed_dim: Optional[int] = None,
        pooling: Literal["mean", "max"] = "mean",
        max_group_size: int = 7,
        rotation_coupling_threshold: int = 3,
        extra_group: Optional[List[Tuple[str, str]]] = None,
        extra_global: Optional[List[str]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_embed_dim = global_embed_dim if global_embed_dim is not None else embed_dim
        self.pooling = pooling
        self.max_group_size = max_group_size
        self.rotation_coupling_threshold = rotation_coupling_threshold

        # Store configurable feature lists (use defaults when None)
        self.extra_group: List[Tuple[str, str]] = (
            list(extra_group) if extra_group is not None else list(DEFAULT_EXTRA_GROUP)
        )
        self.extra_global: List[str] = (
            list(extra_global) if extra_global is not None else list(DEFAULT_EXTRA_GLOBAL)
        )

        # Validate aggregations
        for key, agg in self.extra_group:
            if agg not in _VALID_AGGS:
                raise ValueError(
                    f"Invalid aggregation '{agg}' for extra_group key '{key}'. "
                    f"Must be one of {_VALID_AGGS}."
                )

        # Dynamic scalar dimension
        self._scalar_dim = _BASE_SCALAR_DIM + len(self.extra_group) + len(self.extra_global)

        # Output dimension: h_g + h_glob + scalar_dim
        self._output_dim = self.embed_dim + self.global_embed_dim + self._scalar_dim

    @property
    def output_dim(self) -> int:
        """Dimension of output group feature vectors."""
        return self._output_dim

    @property
    def scalar_dim(self) -> int:
        """Number of scalar features per group (5 + extra_group + extra_global)."""
        return self._scalar_dim

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
        Build feature vectors for all candidate groups.

        Args:
            h: Per-robot embeddings of shape (N, d) or (B, N, d).
                If batched, uses batch dimension 0.
            groups: List of M groups, each group is a list of robot indices.
            h_glob: Global embedding of shape (dg,) or (B, dg).
                If None, computed as mean of h.
            attn_rr: Robot-robot attention weights of shape (N, N) or (B, N, N).
            attn_ro: Robot-obstacle attention weights of shape (N, N_obs) or (B, N, N_obs).
            extra: Dictionary of extra feature tensors.  Must contain
                every key referenced by ``self.extra_group`` (per-robot ``(N,)``
                tensors) and ``self.extra_global`` (global ``(1,)`` scalars).

        Returns:
            X: Group feature matrix of shape (M, D) where D = output_dim.
        """
        # Handle batched vs unbatched input
        if h.dim() == 2:
            h = h.unsqueeze(0)  # (N, d) -> (1, N, d)
        h = h[0]  # (N, d)

        if h_glob is None:
            h_glob = h.mean(dim=0)  # (d,)
        elif h_glob.dim() == 2:
            h_glob = h_glob[0]

        if attn_rr is not None and attn_rr.dim() == 3:
            attn_rr = attn_rr[0]
        if attn_ro is not None and attn_ro.dim() == 3:
            attn_ro = attn_ro[0]

        # Squeeze batch dim from extra tensors
        if extra is not None:
            extra = {k: v[0] if v.dim() == 2 else v for k, v in extra.items()}

        device = h.device
        n_robots = h.shape[0]

        # Build features for each group
        group_features = []
        for group in groups:
            feat = self._build_single_group_feature(
                h=h,
                h_glob=h_glob,
                group=group,
                n_robots=n_robots,
                attn_rr=attn_rr,
                attn_ro=attn_ro,
                extra=extra,
                device=device,
            )
            group_features.append(feat)

        X = torch.stack(group_features, dim=0)  # (M, D)
        return X

    def _build_single_group_feature(
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
        """Build feature vector for a single group."""
        dtype = h.dtype
        group_indices = torch.tensor(group, device=device, dtype=torch.long)
        group_size = len(group)

        # ── Embedding features ──
        h_group = h[group_indices]  # (|g|, d)
        if self.pooling == "mean":
            h_g = h_group.mean(dim=0)
        elif self.pooling == "max":
            h_g = h_group.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # ── Base scalars (always-on, features 1–5) ──
        scalars: List[float] = []

        # 1. size_feat
        scalars.append(group_size / self.max_group_size)

        # 2. coupling_mode
        scalars.append(1.0 if group_size > self.rotation_coupling_threshold else 0.0)

        # 3-5. attention stats [A_in, A_out, A_obs]
        a_in, a_out, a_obs = self._compute_attention_stats(
            group=group, n_robots=n_robots,
            attn_rr=attn_rr, attn_ro=attn_ro,
        )
        scalars.extend([a_in, a_out, a_obs])

        # ── extra_group scalars (aggregated over group members) ──
        group_extras = self._compute_group_extras(extra, group_indices)
        scalars.extend(group_extras)

        # ── extra_global scalars (same value for every group) ──
        global_extras = self._compute_global_extras(extra)
        scalars.extend(global_extras)

        scalar_t = torch.tensor(scalars, device=device, dtype=dtype)  # (scalar_dim,)

        return torch.cat([h_g, h_glob, scalar_t], dim=0)  # (d + dg + scalar_dim,)

    # ------------------------------------------------------------------
    # Separate computation helpers for extra features
    # ------------------------------------------------------------------

    def _compute_group_extras(
        self,
        extra: Optional[Dict[str, torch.Tensor]],
        group_indices: torch.Tensor,
    ) -> List[float]:
        """
        Compute per-group extra scalars by aggregating per-robot tensors
        over the group members.

        Returns a list of floats with length ``len(self.extra_group)``.
        """
        if not self.extra_group:
            return []

        if extra is None:
            return [0.0] * len(self.extra_group)

        result: List[float] = []
        for key, agg in self.extra_group:
            vals = extra[key][group_indices]  # (|g|,)
            if agg == "mean":
                result.append(vals.mean().item())
            elif agg == "min":
                result.append(vals.min().item())
            elif agg == "max":
                result.append(vals.max().item())
            elif agg == "sum":
                result.append(vals.sum().item())
        return result

    def _compute_global_extras(
        self,
        extra: Optional[Dict[str, torch.Tensor]],
    ) -> List[float]:
        """
        Extract global extra scalars from the ``extra`` dict.

        Each key in ``self.extra_global`` should map to a ``(1,)`` tensor.
        Returns a list of floats with length ``len(self.extra_global)``.
        """
        if not self.extra_global:
            return []

        if extra is None:
            return [0.0] * len(self.extra_global)

        result: List[float] = []
        for key in self.extra_global:
            result.append(extra[key].item())
        return result

    @staticmethod
    def _compute_attention_stats(
        group: List[int],
        n_robots: int,
        attn_rr: Optional[torch.Tensor],
        attn_ro: Optional[torch.Tensor],
    ) -> Tuple[float, float, float]:
        """
        Compute attention statistics for a group.

        Returns:
            (A_in, A_out, A_obs)
            - A_in:  Mean attention within group (excluding self)
            - A_out: Mean attention from group to outside robots
            - A_obs: Mean attention from group to obstacles
        """
        if attn_rr is None:
            return (0.0, 0.0, 0.0)

        group_set = set(group)
        outside_indices = [j for j in range(n_robots) if j not in group_set]

        # A_in
        if len(group) > 1:
            a_in_sum = sum(
                attn_rr[i, j].item() for i in group for j in group if i != j
            )
            A_in = a_in_sum / (len(group) * (len(group) - 1))
        else:
            A_in = 0.0

        # A_out
        if outside_indices:
            a_out_sum = sum(
                attn_rr[i, j].item() for i in group for j in outside_indices
            )
            A_out = a_out_sum / (len(group) * len(outside_indices))
        else:
            A_out = 0.0

        # A_obs
        if attn_ro is not None:
            n_obs = attn_ro.shape[1]
            a_obs_sum = sum(
                attn_ro[i, o].item() for i in group for o in range(n_obs)
            )
            A_obs = a_obs_sum / max(len(group) * n_obs, 1)
        else:
            A_obs = 0.0

        return (A_in, A_out, A_obs)
