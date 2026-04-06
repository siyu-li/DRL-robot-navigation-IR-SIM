"""
Attention-Based Group Pooling Module.

Replaces mean/max pooling with a learned 2-head MLP attention mechanism that
summarizes a variable-size set of robot embeddings into a fixed-size group
embedding.  One head can specialise on straggler robots (high distance-to-goal),
the other on coordination / relative features — without any hand-crafted scalars.

Architecture per group (m = number of robots in the group):

    Input: h_group ∈ R^(m, embed_dim)

    The embedding is first split into two semantic parts that reflect the
    GAT backbone's decoder structure (``att_embedding = concat([self_embed,
    attn_out_flat])``):

        h_self  = h_group[:, :split_at]            # (m, split_at)           — robot self-state
        h_neigh = h_group[:, split_at:]            # (m, embed_dim-split_at) — neighbor aggregation

    For each part p ∈ {h_self, h_neigh} and head k ∈ {0, …, n_heads-1}:
        scores_k_p = score_MLP_k_p(h_p)           # (m, 1)
        alphas_k_p = softmax(scores_k_p, dim=0)   # (m, 1) — sums to 1 over m
        head_k_p   = Σ  alphas_k_p * h_p          # (part_dim,) — weighted sum

    Fuse (n_heads*split_at + n_heads*(embed_dim-split_at) = n_heads*embed_dim):
        concat  = [head_0_self ‖ head_1_self ‖ head_0_neigh ‖ head_1_neigh]
        output  = LayerNorm(Linear(n_heads*embed_dim → embed_dim))   # (embed_dim,)

The output dimension equals ``embed_dim`` regardless of ``n_heads``, so the
downstream feature layout ``[h_g_attn ‖ h_glob]`` has shape
``(2 * embed_dim,)`` — identical to the existing embedding-tower input format.

Usage
-----
::

    pool = AttentionGroupPooling(embed_dim=512, n_heads=2)

    # Single group:  (m, 512) → (512,)
    h_g = pool(h_group)

    # All groups at once:  (N, 512) + List[List[int]] → (M, 512)
    H_g = pool.forward_groups(h, groups)

    # Inspect attention:  (m, 512) → (n_heads, m)
    weights = pool.get_attention_weights(h_group)
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_score_mlp(embed_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    """Build a small MLP: embed_dim → hidden[0] → … → hidden[-1] → 1."""
    layers: List[nn.Module] = []
    in_dim = embed_dim
    for h in hidden:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.GELU())
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


class AttentionGroupPooling(nn.Module):
    """
    Multi-head MLP attention pooling over robot embeddings within a group.

    The per-robot embedding is split into two semantic parts corresponding to
    the GAT backbone's decoder output structure:
      - **Self part** (``h[:, :split_at]``): robot's own node-encoder state.
      - **Neighbor part** (``h[:, split_at:]``): aggregated message from
        neighboring robots and obstacles.

    ``n_heads`` independent attention heads are applied to *each part
    separately*, allowing self-state heads to specialise on individual robot
    properties (e.g. straggler detection) while neighbor heads specialise on
    communication-graph structure.  All head outputs are concatenated and
    projected back to ``embed_dim``.

    Args:
        embed_dim: Total per-robot embedding dimension (default 512).
        n_heads: Number of attention heads *per part* (default 2, giving
            2*n_heads heads total).
        score_hidden: Hidden layer widths for each score MLP.
            E.g. ``(128, 64)`` produces MLP: part_dim → 128 → 64 → 1.
        dropout: Dropout applied after the fusion projection (default 0.0).
        split_at: Index at which to split ``embed_dim`` into self vs. neighbor
            parts.  Defaults to ``embed_dim // 2``.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_heads: int = 2,
        score_hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
        split_at: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.score_hidden = tuple(score_hidden)
        self.split_at = split_at if split_at is not None else embed_dim // 2
        self._part2_dim = embed_dim - self.split_at

        # n_heads score MLPs for the self (node-encoder) part
        self.score_mlps_self = nn.ModuleList(
            [_build_score_mlp(self.split_at, self.score_hidden) for _ in range(n_heads)]
        )
        # n_heads score MLPs for the neighbor-aggregation part
        self.score_mlps_neigh = nn.ModuleList(
            [_build_score_mlp(self._part2_dim, self.score_hidden) for _ in range(n_heads)]
        )

        # Fuse all head outputs:
        #   n_heads * split_at  +  n_heads * part2_dim  =  n_heads * embed_dim → embed_dim
        self.fusion_proj = nn.Linear(n_heads * embed_dim, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Output dimension (always equals embed_dim)."""
        return self.embed_dim

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self):
        for mlp_list in (self.score_mlps_self, self.score_mlps_neigh):
            for mlp in mlp_list:
                for m in mlp.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

    # ------------------------------------------------------------------
    # Single-group forward
    # ------------------------------------------------------------------

    def forward(self, h_group: torch.Tensor) -> torch.Tensor:
        """
        Pool a single group of robot embeddings.

        Args:
            h_group: ``(m, embed_dim)`` — embeddings of the m robots in the group.

        Returns:
            ``(embed_dim,)`` group embedding.
        """
        h_self  = h_group[:, :self.split_at]   # (m, split_at)
        h_neigh = h_group[:, self.split_at:]   # (m, part2_dim)

        head_outputs: List[torch.Tensor] = []
        for mlp in self.score_mlps_self:
            scores = mlp(h_self)                            # (m, 1)
            alphas = F.softmax(scores, dim=0)               # (m, 1)
            head_outputs.append((alphas * h_self).sum(dim=0))   # (split_at,)

        for mlp in self.score_mlps_neigh:
            scores = mlp(h_neigh)                           # (m, 1)
            alphas = F.softmax(scores, dim=0)               # (m, 1)
            head_outputs.append((alphas * h_neigh).sum(dim=0))  # (part2_dim,)

        concat = torch.cat(head_outputs, dim=-1)            # (n_heads * embed_dim,)
        out = self.fusion_norm(self.fusion_proj(concat))    # (embed_dim,)
        return self.dropout(out)

    # ------------------------------------------------------------------
    # Batched forward over all groups
    # ------------------------------------------------------------------

    def forward_groups(
        self,
        h: torch.Tensor,
        groups: List[List[int]],
    ) -> torch.Tensor:
        """
        Pool all candidate groups in a single vectorized pass.

        Uses padded index tensors and masking so that variable-size groups
        can be processed as a batch without Python-level loops over groups.
        Masked (padding) positions are assigned score ``-inf`` before softmax
        so they contribute zero attention weight.

        Args:
            h: ``(N, embed_dim)`` per-robot embeddings for all N robots.
            groups: List of M groups, each a list of robot indices.

        Returns:
            ``(M, embed_dim)`` group embeddings — one per candidate group.
        """
        device = h.device
        dtype = h.dtype
        M = len(groups)
        max_gs = max(len(g) for g in groups)

        # --- Build padded index tensor and mask ---
        gi_padded = torch.zeros(M, max_gs, device=device, dtype=torch.long)
        pad_mask = torch.zeros(M, max_gs, device=device, dtype=dtype)  # 1 = real

        for m, group in enumerate(groups):
            gs = len(group)
            gi_padded[m, :gs] = torch.tensor(group, device=device, dtype=torch.long)
            pad_mask[m, :gs] = 1.0

        # h_gathered: (M, max_gs, embed_dim)
        h_gathered = h[gi_padded]

        # inf_mask: (M, max_gs, 1) — -inf for padding positions
        inf_mask = (1.0 - pad_mask).unsqueeze(-1) * (-1e9)

        h_self_g  = h_gathered[..., :self.split_at]   # (M, max_gs, split_at)
        h_neigh_g = h_gathered[..., self.split_at:]   # (M, max_gs, part2_dim)

        head_outputs: List[torch.Tensor] = []
        for mlp in self.score_mlps_self:
            scores = mlp(h_self_g) + inf_mask           # (M, max_gs, 1)
            alphas = F.softmax(scores, dim=1)           # (M, max_gs, 1)
            head_outputs.append((alphas * h_self_g).sum(dim=1))   # (M, split_at)

        for mlp in self.score_mlps_neigh:
            scores = mlp(h_neigh_g) + inf_mask          # (M, max_gs, 1)
            alphas = F.softmax(scores, dim=1)           # (M, max_gs, 1)
            head_outputs.append((alphas * h_neigh_g).sum(dim=1))  # (M, part2_dim)

        # concat: (M, n_heads * embed_dim)
        concat = torch.cat(head_outputs, dim=-1)
        out = self.fusion_norm(self.fusion_proj(concat))   # (M, embed_dim)
        return self.dropout(out)

    # ------------------------------------------------------------------
    # Interpretability: attention weights
    # ------------------------------------------------------------------

    def get_attention_weights(
        self, h_group: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return per-head attention weights for both parts of a single group.

        Useful for visualising which robots each head focuses on.

        Args:
            h_group: ``(m, embed_dim)`` robot embeddings.

        Returns:
            Tuple ``(weights_self, weights_neigh)`` where each tensor has
            shape ``(n_heads, m)`` and each row sums to 1.
        """
        h_self  = h_group[:, :self.split_at]
        h_neigh = h_group[:, self.split_at:]

        def _weights(mlp_list: nn.ModuleList, h_part: torch.Tensor) -> torch.Tensor:
            ws: List[torch.Tensor] = []
            for mlp in mlp_list:
                alphas = F.softmax(mlp(h_part), dim=0)  # (m, 1)
                ws.append(alphas.squeeze(-1))            # (m,)
            return torch.stack(ws, dim=0)               # (n_heads, m)

        return _weights(self.score_mlps_self, h_self), _weights(self.score_mlps_neigh, h_neigh)
