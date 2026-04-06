"""
Attention-based group feature builder for the supervised group switcher.

Replaces the scalar-heavy ``GroupFeatureBuilder`` with a clean, dedicated class
that uses ``AttentionGroupPooling`` as the per-group pooling mechanism.  No
hand-crafted scalars are computed — the attention module learns to summarise
each group's robot embeddings.

Output layout (per group):
    [h_g_attn (embed_dim) ‖ h_glob (embed_dim)]   →   (2 * embed_dim,)

This is a **drop-in replacement** for ``GroupFeatureBuilder``: it accepts the
same ``forward()`` signature.  Arguments ``attn_rr``, ``attn_ro``, and
``extra`` are accepted for API compatibility but are intentionally ignored —
all group information is captured by the attention module.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from robot_nav.models.MARL.switcher.attention_pooling import AttentionGroupPooling


class AttnGroupFeatureBuilder(nn.Module):
    """
    Per-group feature builder based on attention pooling (supervised variant).

    Replaces mean/max pooling + hand-crafted scalars with a learned
    ``AttentionGroupPooling`` module.  The output for each candidate group is:

        x_k = [h_g_attn_k  ‖  h_glob]   ∈  R^(2 * embed_dim)

    where ``h_g_attn_k`` is produced by ``AttentionGroupPooling.forward_groups``
    and ``h_glob`` is the mean-pool of all robot embeddings.

    Args:
        attn_pool: Instantiated ``AttentionGroupPooling`` module (required).
            Its ``embed_dim`` must match ``embed_dim``.
        embed_dim: Dimension of per-robot GAT embeddings (default 512).
        global_embed_dim: Dimension of the global embedding.  If *None*,
            defaults to ``embed_dim``.
    """

    def __init__(
        self,
        attn_pool: AttentionGroupPooling,
        embed_dim: int = 512,
        global_embed_dim: Optional[int] = None,
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

    # ------------------------------------------------------------------
    # Dimension properties  (mirrors GroupFeatureBuilder API)
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Total feature vector dimension: embed_dim + global_embed_dim."""
        return self.embed_dim + self.global_embed_dim

    @property
    def scalar_dim(self) -> int:
        """No scalars — always 0."""
        return 0

    # ------------------------------------------------------------------
    # Forward  (drop-in replacement for GroupFeatureBuilder.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        groups: List[List[int]],
        h_glob: Optional[torch.Tensor] = None,
        attn_rr: Optional[torch.Tensor] = None,   # accepted, unused
        attn_ro: Optional[torch.Tensor] = None,   # accepted, unused
        extra: Optional[Dict[str, torch.Tensor]] = None,  # accepted, unused
    ) -> torch.Tensor:
        """
        Build attention-pooled feature matrix for all candidate groups.

        Args:
            h: Per-robot embeddings ``(N, embed_dim)`` or ``(1, N, embed_dim)``.
            groups: List of M groups, each a list of robot indices.
            h_glob: Global embedding ``(global_embed_dim,)`` or
                ``(1, global_embed_dim)``.  If *None*, computed as
                ``h.mean(dim=0)``.
            attn_rr: Ignored (accepted for API compatibility).
            attn_ro: Ignored (accepted for API compatibility).
            extra: Ignored (accepted for API compatibility).

        Returns:
            X: ``(M, 2 * embed_dim)`` group feature matrix.
                Layout per row: ``[h_g_attn ‖ h_glob]``.
        """
        # --- Squeeze batch dimension ---
        if h.dim() == 3:
            h = h[0]
        if h_glob is None:
            h_glob = h.mean(dim=0)
        elif h_glob.dim() == 2:
            h_glob = h_glob[0]

        M = len(groups)

        # --- Per-group embeddings via attention pooling: (M, embed_dim) ---
        h_g = self.attn_pool.forward_groups(h, groups)

        # --- Broadcast global embedding: (M, global_embed_dim) ---
        h_glob_exp = h_glob.unsqueeze(0).expand(M, -1)

        # --- Concatenate: (M, embed_dim + global_embed_dim) ---
        return torch.cat([h_g, h_glob_exp], dim=1)
