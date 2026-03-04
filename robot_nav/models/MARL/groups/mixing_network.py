"""
Mixing Network for Group-Based Velocity Coupling.

A small shared network ``f_lin(e_i) → scalar`` that maps a frozen per-robot
GAT embedding to a mixing weight.  One set of parameters is shared across all
robots.

Architecture (intentionally tiny — only this network is trained):
    embedding_dim → 128 → 64 → 1

The output is a raw logit; the caller applies softmax over the active group
members (after masking arrived robots to −∞).

Training note
-------------
The GAT + TD3 policy is frozen.  Only the parameters of this network receive
gradients.
"""

import torch
import torch.nn as nn


class MixingNetwork(nn.Module):
    """
    Shared mixing-weight network:  f_lin(e_i) → scalar logit.

    Args:
        embedding_dim (int): Dimensionality of the per-robot GAT embedding
            (output of the actor's attention + decode, i.e. ``embedding_dim * 2``
            from ``AttentionObstacleOptimized``).  Default 512 (= 256 * 2).
        hidden_dim (int): Hidden layer width.  Default 128.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute raw mixing logits for a set of embeddings.

        Args:
            embeddings (Tensor): Shape ``(K, embedding_dim)`` where *K* is the
                number of robots in the current group.

        Returns:
            Tensor: Raw logits of shape ``(K,)``.
        """
        return self.net(embeddings).squeeze(-1)
