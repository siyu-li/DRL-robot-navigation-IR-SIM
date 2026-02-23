"""
Ranking Losses for Training Group Switcher.

Implements pairwise ranking losses for learning to rank groups.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def pairwise_logistic_ranking_loss(
    logits: torch.Tensor,
    pairs: torch.LongTensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss (RankNet-style).
    
    For each pair (pos_idx, neg_idx), the loss is:
        softplus(-(logits[pos] - logits[neg]))
    
    This encourages logits[pos] > logits[neg].
    
    Args:
        logits: Scores for each group, shape (M,).
        pairs: Pairwise constraints, shape (K, 2).
            pairs[k] = [pos_idx, neg_idx] means pos should rank higher than neg.
        reduction: How to reduce over pairs ("mean", "sum", "none").
        
    Returns:
        Loss value (scalar if reduction != "none", else shape (K,)).
    """
    if pairs.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    pos_indices = pairs[:, 0]
    neg_indices = pairs[:, 1]
    
    pos_logits = logits[pos_indices]
    neg_logits = logits[neg_indices]
    
    # softplus(-(pos - neg)) = log(1 + exp(neg - pos))
    # When pos >> neg, loss -> 0
    # When neg >> pos, loss -> (neg - pos)
    diff = pos_logits - neg_logits
    loss = F.softplus(-diff)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def hinge_ranking_loss(
    logits: torch.Tensor,
    pairs: torch.LongTensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Pairwise hinge ranking loss.
    
    For each pair (pos_idx, neg_idx), the loss is:
        max(0, margin - (logits[pos] - logits[neg]))
    
    Args:
        logits: Scores for each group, shape (M,).
        pairs: Pairwise constraints, shape (K, 2).
        margin: Margin for hinge loss.
        reduction: How to reduce over pairs.
        
    Returns:
        Loss value.
    """
    if pairs.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    pos_indices = pairs[:, 0]
    neg_indices = pairs[:, 1]
    
    pos_logits = logits[pos_indices]
    neg_logits = logits[neg_indices]
    
    diff = pos_logits - neg_logits
    loss = F.relu(margin - diff)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def listwise_softmax_loss(
    logits: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    """
    Listwise softmax cross-entropy loss.
    
    Treats group selection as multi-class classification where
    target_idx is the correct class.
    
    Args:
        logits: Scores for each group, shape (M,).
        target_idx: Index of the target (best) group.
        
    Returns:
        Cross-entropy loss (scalar).
    """
    target = torch.tensor([target_idx], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits.unsqueeze(0), target)


def build_pairs_from_scores(
    scores: torch.Tensor,
    margin: float = 0.0,
) -> torch.LongTensor:
    """
    Build all valid ranking pairs from ground-truth scores.
    
    Creates pairs (i, j) where scores[i] > scores[j] + margin.
    
    Args:
        scores: Ground-truth scores for each group, shape (M,).
        margin: Minimum score difference to create a pair.
        
    Returns:
        Pairs tensor of shape (K, 2), where K is number of valid pairs.
    """
    device = scores.device
    n = len(scores)
    
    pairs = []
    for i in range(n):
        for j in range(n):
            if scores[i] > scores[j] + margin:
                pairs.append([i, j])
    
    if len(pairs) == 0:
        return torch.zeros(0, 2, device=device, dtype=torch.long)
    
    return torch.tensor(pairs, device=device, dtype=torch.long)


def build_pairs_from_ranking(
    ranking: List[int],
) -> torch.LongTensor:
    """
    Build ranking pairs from an ordered list of indices.
    
    ranking[0] should be the best group, ranking[-1] the worst.
    Creates pairs (ranking[i], ranking[j]) for all i < j.
    
    Args:
        ranking: Ordered list of group indices (best to worst).
        
    Returns:
        Pairs tensor of shape (K, 2).
    """
    pairs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            # ranking[i] should rank higher than ranking[j]
            pairs.append([ranking[i], ranking[j]])
    
    if len(pairs) == 0:
        return torch.zeros(0, 2, dtype=torch.long)
    
    return torch.tensor(pairs, dtype=torch.long)


def compute_ranking_accuracy(
    logits: torch.Tensor,
    pairs: torch.LongTensor,
) -> float:
    """
    Compute pairwise ranking accuracy.
    
    Accuracy = fraction of pairs where logits[pos] > logits[neg].
    
    Args:
        logits: Predicted scores, shape (M,).
        pairs: Ground-truth pairs, shape (K, 2).
        
    Returns:
        Accuracy in [0, 1].
    """
    if pairs.numel() == 0:
        return 1.0
    
    pos_indices = pairs[:, 0]
    neg_indices = pairs[:, 1]
    
    correct = (logits[pos_indices] > logits[neg_indices]).float()
    return correct.mean().item()


def compute_top1_accuracy(
    logits: torch.Tensor,
    target_idx: int,
) -> bool:
    """
    Check if the top-ranked group matches the target.
    
    Args:
        logits: Predicted scores, shape (M,).
        target_idx: Index of the correct best group.
        
    Returns:
        True if argmax(logits) == target_idx.
    """
    return logits.argmax().item() == target_idx


class RankingLossWithScheduledMargin:
    """
    Hinge ranking loss with scheduled margin that increases during training.
    
    Starts with small margin (easy) and increases to target margin (hard).
    """
    
    def __init__(
        self,
        initial_margin: float = 0.1,
        final_margin: float = 1.0,
        warmup_steps: int = 1000,
    ):
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def __call__(
        self,
        logits: torch.Tensor,
        pairs: torch.LongTensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute loss with current margin."""
        margin = self.get_current_margin()
        return hinge_ranking_loss(logits, pairs, margin=margin, reduction=reduction)
    
    def get_current_margin(self) -> float:
        """Get margin for current step."""
        if self.current_step >= self.warmup_steps:
            return self.final_margin
        
        ratio = self.current_step / self.warmup_steps
        return self.initial_margin + ratio * (self.final_margin - self.initial_margin)
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
