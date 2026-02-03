"""
GroupSwitcher Neural Network for Ranking-Based Group Selection.

Scores candidate groups and selects the best one based on learned ranking.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupSwitcher(nn.Module):
    """
    Two-tower fusion network that scores candidate groups for selection.
    
    Takes group feature vectors and outputs logits for ranking.
    
    Architecture (Two-Tower Fusion):
        Tower 1 (embeddings): e = [h_g || h_glob] ∈ R^1024
            Linear(1024 → 256) → GELU → LayerNorm
        
        Tower 2 (scalars): s ∈ R^4 (size_feat + attn_stats)
            Linear(4 → 32) → GELU → LayerNorm
        
        Fusion: [e' || s'] ∈ R^288
            Linear(288 → 256) → GELU → LayerNorm → Dropout
            Linear(256 → 1)
    
    Args:
        embed_dim: Dimension of per-robot embeddings (d). Default 512.
        scalar_dim: Dimension of scalar features (size + attn_stats). Default 4.
        embed_hidden: Hidden dimension for embedding tower. Default 256.
        scalar_hidden: Hidden dimension for scalar tower. Default 32.
        fusion_hidden: Hidden dimension for fusion layer. Default 256.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        scalar_dim: int = 4,
        embed_hidden: int = 256,
        scalar_hidden: int = 32,
        fusion_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.scalar_dim = scalar_dim
        self.embed_hidden = embed_hidden
        self.scalar_hidden = scalar_hidden
        self.fusion_hidden = fusion_hidden
        self.dropout_rate = dropout
        
        # For backward compatibility: compute expected input_dim
        # input_dim = h_g (embed_dim) + h_glob (embed_dim) + scalars (scalar_dim)
        self.input_dim = 2 * embed_dim + scalar_dim
        
        # Tower 1: Embedding tower
        # Input: [h_g || h_glob] ∈ R^(2*embed_dim)
        self.embed_tower = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_hidden),
            nn.GELU(),
            nn.LayerNorm(embed_hidden),
        )
        
        # Tower 2: Scalar tower
        # Input: [size_feat, A_in, A_out, A_obs] ∈ R^scalar_dim
        self.scalar_tower = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden),
            nn.GELU(),
            nn.LayerNorm(scalar_hidden),
        )
        
        # Fusion: [e' || s'] ∈ R^(embed_hidden + scalar_hidden)
        fusion_input_dim = embed_hidden + scalar_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for GELU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # GELU approximates ReLU, use gain ~1.0
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for all candidate groups using two-tower fusion.
        
        Args:
            X: Group feature matrix of shape (M, D) where
               D = 2*embed_dim + scalar_dim
               Layout: [h_g (embed_dim), h_glob (embed_dim), scalars (scalar_dim)]
            
        Returns:
            logits: Scores for each group, shape (M,).
        """
        # Split input into embeddings and scalars
        # X layout: [h_g, h_glob, size_feat, A_in, A_out, A_obs, ...]
        embed_features = X[:, :2 * self.embed_dim]  # [h_g || h_glob]
        scalar_features = X[:, 2 * self.embed_dim:]  # [size_feat, attn_stats, extras]
        
        # Tower 1: Process embeddings
        e_prime = self.embed_tower(embed_features)  # (M, embed_hidden)
        
        # Tower 2: Process scalars
        s_prime = self.scalar_tower(scalar_features)  # (M, scalar_hidden)
        
        # Fusion
        fused = torch.cat([e_prime, s_prime], dim=-1)  # (M, embed_hidden + scalar_hidden)
        logits = self.fusion(fused).squeeze(-1)  # (M,)
        
        return logits
    
    def select_group(
        self,
        logits: torch.Tensor,
        mode: Literal["argmax", "softmax"] = "argmax",
    ) -> int:
        """
        Select a group index based on logits.
        
        Args:
            logits: Scores for each group, shape (M,).
            mode: Selection mode.
                "argmax": Select group with highest logit (deterministic).
                "softmax": Sample from softmax distribution (stochastic).
                
        Returns:
            Selected group index.
        """
        if mode == "argmax":
            return logits.argmax().item()
        elif mode == "softmax":
            probs = F.softmax(logits, dim=0)
            return torch.multinomial(probs, 1).item()
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
    
    def sample_group(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[int, torch.Tensor]:
        """
        Sample a group index using softmax with temperature.
        
        Args:
            logits: Scores for each group, shape (M,).
            temperature: Softmax temperature. Lower = more deterministic.
            
        Returns:
            Tuple of (selected_index, log_probability).
        """
        scaled_logits = logits / max(temperature, 1e-8)
        probs = F.softmax(scaled_logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        log_prob = F.log_softmax(scaled_logits, dim=0)[idx]
        return idx, log_prob
    
    def get_selection_probs(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get selection probabilities for all groups.
        
        Args:
            logits: Scores for each group, shape (M,).
            temperature: Softmax temperature.
            
        Returns:
            Probabilities for each group, shape (M,).
        """
        scaled_logits = logits / max(temperature, 1e-8)
        return F.softmax(scaled_logits, dim=0)


class GroupSwitcherWithBaseline(nn.Module):
    """
    GroupSwitcher with optional value baseline for variance reduction in RL training.
    
    Adds a separate value head that estimates expected return from current state.
    Uses the same two-tower fusion architecture as GroupSwitcher.
    
    Args:
        embed_dim: Dimension of per-robot embeddings (d). Default 256.
        scalar_dim: Dimension of scalar features. Default 4.
        embed_hidden: Hidden dimension for embedding tower. Default 256.
        scalar_hidden: Hidden dimension for scalar tower. Default 32.
        fusion_hidden: Hidden dimension for fusion layer. Default 256.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        scalar_dim: int = 6,
        embed_hidden: int = 256,
        scalar_hidden: int = 32,
        fusion_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.scorer = GroupSwitcher(
            embed_dim=embed_dim,
            scalar_dim=scalar_dim,
            embed_hidden=embed_hidden,
            scalar_hidden=scalar_hidden,
            fusion_hidden=fusion_hidden,
            dropout=dropout,
        )
        
        # Value baseline network (state-dependent, not group-dependent)
        # Takes mean of group features as state representation
        # Uses same two-tower architecture
        input_dim = 2 * embed_dim + scalar_dim
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden),
            nn.Linear(fusion_hidden, 1),
        )
    
    def forward(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits and value baseline.
        
        Args:
            X: Group feature matrix of shape (M, D).
            
        Returns:
            Tuple of:
                logits: Scores for each group, shape (M,).
                value: Baseline value estimate, shape (1,).
        """
        logits = self.scorer(X)
        
        # Use mean of group features as state representation for value
        state_repr = X.mean(dim=0, keepdim=True)  # (1, D)
        value = self.value_head(state_repr).squeeze()  # scalar
        
        return logits, value
    
    def select_group(
        self,
        logits: torch.Tensor,
        mode: Literal["argmax", "softmax"] = "argmax",
    ) -> int:
        """Select a group index based on logits."""
        return self.scorer.select_group(logits, mode)
    
    def sample_group(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[int, torch.Tensor]:
        """Sample a group index using softmax with temperature."""
        return self.scorer.sample_group(logits, temperature)
