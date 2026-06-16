"""
SwitcherHead: Trainable MLP that maps the mean-pooled pre-decoder embedding
to a binary {coarse, precise} logit pair.

This module is the only trainable component in the CAPSwitcher system;
the GAT backbone that produces the input embeddings is fully frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwitcherHead(nn.Module):
    """
    CAPSwitcher policy head (trainable).

    Maps the mean-pooled swarm embedding (B, embed_dim) to raw logits
    (B, 2) for a Categorical distribution over {0=coarse, 1=precise}.

    Architecture:
        Linear(embed_dim → hidden1) → ReLU
        Linear(hidden1   → hidden2) → ReLU
        Linear(hidden2   → 2)

    Args:
        embed_dim: Dimension of the input embedding. Default 512.
        hidden1:   First hidden layer width.  Default 256.
        hidden2:   Second hidden layer width. Default 128.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden1: int = 256,
        hidden2: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim) mean-pooled swarm embedding.

        Returns:
            logits: (B, 2) raw logits — feed to ``torch.distributions.Categorical``.
        """
        return self.net(x)
