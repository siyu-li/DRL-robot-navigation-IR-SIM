"""
Group Feature Builder for Ranking-Based Group Switcher.

Constructs feature vectors for each candidate group from:
- Per-robot embeddings from GAT backbone
- Global embedding (optional)
- Attention statistics (optional)
- Extra per-robot features like distance-to-goal, clearance (optional)
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn


class GroupFeatureBuilder(nn.Module):
    """
    Builds feature vectors for candidate groups from per-robot embeddings and optional attention info.
    
    Output feature format per group:
        [h_g, h_glob, size_feat, attn_stats, extra_stats]
    
    Where:
        - h_g: Pooled group embedding (d-dim)
        - h_glob: Global embedding (dg-dim, defaults to mean of h if not provided)
        - size_feat: Normalized group size (1-dim)
        - attn_stats: [A_in, A_out, A_obs] if attention available, else zeros (3-dim)
        - extra_stats: Additional per-robot feature statistics (variable dim)
    
    Args:
        embed_dim: Dimension of per-robot embeddings (d).
        global_embed_dim: Dimension of global embedding (dg). If None, uses embed_dim.
        pooling: Pooling method for group embedding ("mean" or "max").
        extra_feature_names: List of extra feature names to include (e.g., ["dist_to_goal", "clearance"]).
        extra_aggregations: Aggregation methods for each extra feature ("mean", "min", "max").
            If single value, applies to all extra features.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        global_embed_dim: Optional[int] = None,
        pooling: Literal["mean", "max"] = "mean",
        extra_feature_names: Optional[List[str]] = None,
        extra_aggregations: Optional[List[Literal["mean", "min", "max"]]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_embed_dim = global_embed_dim if global_embed_dim is not None else embed_dim
        self.pooling = pooling
        
        # Extra features configuration
        self.extra_feature_names = extra_feature_names or []
        if extra_aggregations is None:
            # Default: use mean for all
            self.extra_aggregations = ["mean"] * len(self.extra_feature_names)
        elif isinstance(extra_aggregations, str):
            self.extra_aggregations = [extra_aggregations] * len(self.extra_feature_names)
        else:
            self.extra_aggregations = list(extra_aggregations)
        
        assert len(self.extra_aggregations) == len(self.extra_feature_names), \
            "Must provide one aggregation per extra feature"
        
        # Compute output dimension
        self._output_dim = (
            self.embed_dim +          # h_g
            self.global_embed_dim +   # h_glob
            1 +                       # size_feat
            3 +                       # attn_stats [A_in, A_out, A_obs]
            len(self.extra_feature_names)  # extra_stats
        )
    
    @property
    def output_dim(self) -> int:
        """Dimension of output group feature vectors."""
        return self._output_dim
    
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
                From AttentionObstacle: hard_weights_rr or combined weights.
            attn_ro: Robot-obstacle attention weights of shape (N, N_obs) or (B, N, N_obs).
                From AttentionObstacle: hard_weights_ro.
            extra: Dictionary of extra per-robot features.
                Each value should be shape (N,) or (B, N).
                
        Returns:
            X: Group feature matrix of shape (M, D) where D = output_dim.
        """
        # Handle batched vs unbatched input
        if h.dim() == 2:
            h = h.unsqueeze(0)  # (N, d) -> (1, N, d)
        batch_size, n_robots, d = h.shape
        
        # Use first batch element for unbatched API
        h = h[0]  # (N, d)
        
        if h_glob is None:
            h_glob = h.mean(dim=0)  # (d,)
        elif h_glob.dim() == 2:
            h_glob = h_glob[0]  # (dg,)
        
        if attn_rr is not None and attn_rr.dim() == 3:
            attn_rr = attn_rr[0]  # (N, N)
        if attn_ro is not None and attn_ro.dim() == 3:
            attn_ro = attn_ro[0]  # (N, N_obs)
        
        # Process extra features
        if extra is not None:
            extra = {k: v[0] if v.dim() == 2 else v for k, v in extra.items()}
        
        device = h.device
        n_groups = len(groups)
        
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
        group_indices = torch.tensor(group, device=device, dtype=torch.long)
        group_size = len(group)
        
        # 1. Group embedding via pooling
        h_group = h[group_indices]  # (|g|, d)
        if self.pooling == "mean":
            h_g = h_group.mean(dim=0)  # (d,)
        elif self.pooling == "max":
            h_g = h_group.max(dim=0)[0]  # (d,)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # 2. Size feature (normalized by max size=3)
        size_feat = torch.tensor([group_size / 3.0], device=device, dtype=h.dtype)
        
        # 3. Attention statistics
        attn_stats = self._compute_attention_stats(
            group=group,
            n_robots=n_robots,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            device=device,
            dtype=h.dtype,
        )
        
        # 4. Extra feature statistics
        extra_stats = self._compute_extra_stats(
            group=group,
            extra=extra,
            device=device,
            dtype=h.dtype,
        )
        
        # Concatenate all features
        feat = torch.cat([h_g, h_glob, size_feat, attn_stats, extra_stats], dim=0)
        return feat
    
    def _compute_attention_stats(
        self,
        group: List[int],
        n_robots: int,
        attn_rr: Optional[torch.Tensor],
        attn_ro: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute attention statistics for a group.
        
        Returns:
            Tensor of shape (3,): [A_in, A_out, A_obs]
            - A_in: Mean attention from group robots to group robots
            - A_out: Mean attention from group robots to robots outside group
            - A_obs: Mean attention from group robots to obstacles
        """
        if attn_rr is None:
            # No attention info available, return zeros
            return torch.zeros(3, device=device, dtype=dtype)
        
        group_set = set(group)
        outside_set = set(range(n_robots)) - group_set
        group_indices = list(group)
        outside_indices = list(outside_set)
        
        # A_in: attention within group (excluding self-attention)
        if len(group) > 1:
            a_in_sum = 0.0
            a_in_count = 0
            for i in group:
                for j in group:
                    if i != j:
                        a_in_sum += attn_rr[i, j].item()
                        a_in_count += 1
            A_in = a_in_sum / max(a_in_count, 1)
        else:
            A_in = 0.0
        
        # A_out: attention from group to outside
        if len(outside_indices) > 0:
            a_out_sum = 0.0
            a_out_count = 0
            for i in group:
                for j in outside_indices:
                    a_out_sum += attn_rr[i, j].item()
                    a_out_count += 1
            A_out = a_out_sum / max(a_out_count, 1)
        else:
            A_out = 0.0
        
        # A_obs: attention to obstacles
        if attn_ro is not None:
            n_obs = attn_ro.shape[1]
            a_obs_sum = 0.0
            a_obs_count = 0
            for i in group:
                for o in range(n_obs):
                    a_obs_sum += attn_ro[i, o].item()
                    a_obs_count += 1
            A_obs = a_obs_sum / max(a_obs_count, 1)
        else:
            A_obs = 0.0
        
        return torch.tensor([A_in, A_out, A_obs], device=device, dtype=dtype)
    
    def _compute_extra_stats(
        self,
        group: List[int],
        extra: Optional[Dict[str, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute aggregated statistics for extra per-robot features."""
        if not self.extra_feature_names:
            return torch.zeros(0, device=device, dtype=dtype)
        
        stats = []
        for name, agg in zip(self.extra_feature_names, self.extra_aggregations):
            if extra is None or name not in extra:
                # Feature not provided, use zero
                stats.append(0.0)
            else:
                values = extra[name][group]  # (|g|,)
                if agg == "mean":
                    stat = values.mean().item()
                elif agg == "min":
                    stat = values.min().item()
                elif agg == "max":
                    stat = values.max().item()
                else:
                    raise ValueError(f"Unknown aggregation: {agg}")
                stats.append(stat)
        
        return torch.tensor(stats, device=device, dtype=dtype)


def compute_attention_stats_vectorized(
    groups: List[List[int]],
    n_robots: int,
    attn_rr: torch.Tensor,
    attn_ro: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Vectorized computation of attention statistics for all groups.
    
    This is more efficient than per-group computation for large group counts.
    
    Args:
        groups: List of M groups.
        n_robots: Total number of robots.
        attn_rr: Robot-robot attention of shape (N, N).
        attn_ro: Robot-obstacle attention of shape (N, N_obs), optional.
        
    Returns:
        Tensor of shape (M, 3): [A_in, A_out, A_obs] for each group.
    """
    device = attn_rr.device
    dtype = attn_rr.dtype
    n_groups = len(groups)
    
    # Create group membership masks
    max_group_size = max(len(g) for g in groups)
    group_masks = torch.zeros(n_groups, n_robots, device=device, dtype=dtype)
    
    for m, group in enumerate(groups):
        for i in group:
            group_masks[m, i] = 1.0
    
    # A_in: within-group attention (excluding diagonal)
    # For each group, sum attn[i,j] where both i,j are in group and i != j
    diag_mask = 1.0 - torch.eye(n_robots, device=device, dtype=dtype)
    attn_no_diag = attn_rr * diag_mask  # (N, N)
    
    # Expand for broadcasting: (M, N, 1) @ (1, N, N) @ (M, 1, N)
    # A_in[m] = sum over i,j in group[m] of attn[i,j]
    A_in = torch.zeros(n_groups, device=device, dtype=dtype)
    A_out = torch.zeros(n_groups, device=device, dtype=dtype)
    
    for m, group in enumerate(groups):
        mask_in = group_masks[m]  # (N,)
        mask_out = 1.0 - mask_in
        
        # Within-group
        in_attn = attn_no_diag * mask_in.unsqueeze(0) * mask_in.unsqueeze(1)
        n_pairs_in = max(len(group) * (len(group) - 1), 1)
        A_in[m] = in_attn.sum() / n_pairs_in
        
        # Out-of-group
        out_attn = attn_rr * mask_in.unsqueeze(0) * mask_out.unsqueeze(1)
        n_pairs_out = max(len(group) * (n_robots - len(group)), 1)
        A_out[m] = out_attn.sum() / n_pairs_out
    
    # A_obs: attention to obstacles
    if attn_ro is not None:
        n_obs = attn_ro.shape[1]
        A_obs = torch.zeros(n_groups, device=device, dtype=dtype)
        for m, group in enumerate(groups):
            obs_attn = attn_ro[group, :].sum() / max(len(group) * n_obs, 1)
            A_obs[m] = obs_attn
    else:
        A_obs = torch.zeros(n_groups, device=device, dtype=dtype)
    
    return torch.stack([A_in, A_out, A_obs], dim=1)  # (M, 3)
