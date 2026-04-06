"""
Attention-based group feature builder for the RL (PPO) group switcher.

Dual-output counterpart of ``AttnGroupFeatureBuilder`` for the actor-critic
architecture:

  1. **Actor input** — ``forward(...)`` → ``(M, 2 * embed_dim)``
     Per-group features produced by ``AttentionGroupPooling``, no scalars.

  2. **Critic input** — ``build_state_features(...)`` → ``(S,)``
     Global state features ``[h_glob ‖ state_scalars]``.
     State scalars are global-level (not per-group heuristics) and are kept
     to give the value head useful context about overall episode progress.

This is a **drop-in replacement** for ``RLFeatureBuilder``: it exposes the
same properties (``group_feature_dim``, ``state_feature_dim``,
``group_scalar_dim``, ``state_scalar_dim``) and the same two method signatures.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from robot_nav.models.MARL.switcher.attention_pooling import AttentionGroupPooling

_VALID_AGGS = {"mean", "min", "max", "sum", "first"}


class RLAttnFeatureBuilder(nn.Module):
    """
    RL feature builder using attention pooling for per-group features.

    Produces two outputs that mirror the ``RLFeatureBuilder`` API:

    * ``forward(h, groups, h_glob, ...)`` → ``(M, 2*embed_dim)``
      Actor input: attention-pooled group embeddings concatenated with the
      global mean-pool embedding.  ``attn_rr``, ``attn_ro``, ``extra`` are
      accepted for API compatibility but ignored for group features.

    * ``build_state_features(h, h_glob, extra)`` → ``(S,)``
      Critic input: ``[h_glob ‖ state_scalars]``.
      State scalars are kept (they describe global episode state, not
      group-level heuristics).

    Args:
        attn_pool: Instantiated ``AttentionGroupPooling`` (required).
            Its ``embed_dim`` must match ``embed_dim``.
        embed_dim: Dimension of per-robot GAT embeddings (default 512).
        global_embed_dim: Dimension of the global embedding.  If *None*,
            defaults to ``embed_dim``.
        state_scalars: State-level scalars for the RL critic
            ``[(key, agg), ...]``.  Aggregated globally over all robots.
    """

    def __init__(
        self,
        attn_pool: AttentionGroupPooling,
        embed_dim: int = 512,
        global_embed_dim: Optional[int] = None,
        state_scalars: List[Tuple[str, str]] = (),
    ):
        super().__init__()

        if attn_pool.embed_dim != embed_dim:
            raise ValueError(
                f"attn_pool.embed_dim ({attn_pool.embed_dim}) must match "
                f"embed_dim ({embed_dim})."
            )

        self.attn_pool = attn_pool
        self.embed_dim = embed_dim
        self.global_embed_dim = global_embed_dim if global_embed_dim is not None else embed_dim
        self._state_scalars: List[Tuple[str, str]] = list(state_scalars)

        # Validate aggregations
        for key, agg in self._state_scalars:
            if agg not in _VALID_AGGS:
                raise ValueError(
                    f"Invalid aggregation '{agg}' for state_scalars key '{key}'. "
                    f"Must be one of {_VALID_AGGS}."
                )

    # ------------------------------------------------------------------
    # Dimension properties  (mirrors RLFeatureBuilder API)
    # ------------------------------------------------------------------

    @property
    def group_scalar_dim(self) -> int:
        """No per-group scalars — always 0."""
        return 0

    @property
    def state_scalar_dim(self) -> int:
        """Number of state-level scalar features for the value head."""
        return len(self._state_scalars)

    @property
    def group_feature_dim(self) -> int:
        """Total dimension of per-group feature vector: 2 * embed_dim."""
        return self.embed_dim + self.global_embed_dim

    @property
    def state_feature_dim(self) -> int:
        """Total dimension of state feature vector: global_embed_dim + state_scalar_dim."""
        return self.global_embed_dim + self.state_scalar_dim

    @classmethod
    def from_config(
        cls,
        cfg,
        attn_pool: AttentionGroupPooling,
        embed_dim: int = 512,
        global_embed_dim: Optional[int] = None,
    ) -> "RLAttnFeatureBuilder":
        """Create an ``RLAttnFeatureBuilder`` from a ``SwitcherScalarConfig``.

        Args:
            cfg: ``SwitcherScalarConfig`` instance.
            attn_pool: Pre-constructed ``AttentionGroupPooling`` module.
            embed_dim, global_embed_dim: Architecture hyper-parameters.
        """
        return cls(
            attn_pool=attn_pool,
            embed_dim=embed_dim,
            global_embed_dim=global_embed_dim,
            state_scalars=cfg.state_scalars,
        )

    # ------------------------------------------------------------------
    # Actor features: per-group embeddings
    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        groups: List[List[int]],
        h_glob: Optional[torch.Tensor] = None,
        attn_rr: Optional[torch.Tensor] = None,   # accepted, unused for groups
        attn_ro: Optional[torch.Tensor] = None,   # accepted, unused for groups
        extra: Optional[Dict[str, torch.Tensor]] = None,  # accepted, unused
    ) -> torch.Tensor:
        """
        Build per-group feature matrix for the actor (policy head).

        Args:
            h: Per-robot embeddings ``(N, embed_dim)`` or ``(1, N, embed_dim)``.
            groups: List of M groups, each a list of robot indices.
            h_glob: Global embedding ``(global_embed_dim,)`` or
                ``(1, global_embed_dim)``.  If *None*, computed as
                ``h.mean(dim=0)``.
            attn_rr: Ignored (API compatibility).
            attn_ro: Ignored (API compatibility).
            extra: Ignored (API compatibility).

        Returns:
            X: ``(M, 2 * embed_dim)`` group feature matrix.
        """
        h, h_glob = self._unpack_h(h, h_glob)
        M = len(groups)

        # Per-group attention pooling: (M, embed_dim)
        h_g = self.attn_pool.forward_groups(h, groups)

        # Broadcast global embedding: (M, global_embed_dim)
        h_glob_exp = h_glob.unsqueeze(0).expand(M, -1)

        return torch.cat([h_g, h_glob_exp], dim=1)  # (M, 2 * embed_dim)

    # ------------------------------------------------------------------
    # Critic features: global state
    # ------------------------------------------------------------------

    def build_state_features(
        self,
        h: torch.Tensor,
        h_glob: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Build state-level feature vector for the critic (value head).

        Layout: ``[h_glob (global_embed_dim) ‖ state_scalars]``

        State scalars are aggregated globally (over all robots), not per-group.
        This path is identical to ``RLFeatureBuilder.build_state_features``.

        Args:
            h: Per-robot embeddings ``(N, d)`` or ``(1, N, d)``.
            h_glob: Global embedding ``(dg,)`` or ``(1, dg)``.
                If *None*, computed as ``h.mean(dim=0)``.
            extra: Dict of per-robot feature tensors.

        Returns:
            state: 1-D tensor ``(S,)`` where S = global_embed_dim + state_scalar_dim.
        """
        h, h_glob = self._unpack_h(h, h_glob)
        if extra is not None:
            extra = {k: (v[0] if v.dim() == 2 else v) for k, v in extra.items()}

        device = h.device
        dtype = h.dtype

        if not self._state_scalars or extra is None:
            scalars = torch.zeros(self.state_scalar_dim, device=device, dtype=dtype)
        else:
            parts: List[torch.Tensor] = []
            for key, agg in self._state_scalars:
                vals = extra[key]
                if agg == "mean":
                    parts.append(vals.mean())
                elif agg == "min":
                    parts.append(vals.min())
                elif agg == "max":
                    parts.append(vals.max())
                elif agg == "sum":
                    parts.append(vals.sum())
                elif agg == "first":
                    parts.append(vals[0])
                else:
                    parts.append(torch.tensor(0.0, device=device, dtype=dtype))
            scalars = torch.stack(parts)

        return torch.cat([h_glob, scalars], dim=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_h(
        self,
        h: torch.Tensor,
        h_glob: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Squeeze batch dims and compute h_glob if not provided."""
        if h.dim() == 3:
            h = h[0]
        if h_glob is None:
            h_glob = h.mean(dim=0)
        elif h_glob.dim() == 2:
            h_glob = h_glob[0]
        return h, h_glob
