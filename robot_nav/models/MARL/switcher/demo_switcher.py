"""
Demo script for Group Switcher module.

Shows usage of:
- GroupFeatureBuilder for constructing group features
- GroupSwitcher network for scoring groups
- Ranking losses for training

Run with:
    python -m robot_nav.models.MARL.switcher.demo_switcher
    
# Default config (embed_dim=256, scalar_dim=4)
switcher = GroupSwitcher()

# Or explicit config
switcher = GroupSwitcher(
    embed_dim=256,      # h_g and h_glob dimension
    scalar_dim=4,       # size_feat (1) + attn_stats (3)
    embed_hidden=256,   # Tower 1 output
    scalar_hidden=32,   # Tower 2 output  
    fusion_hidden=256,  # Fusion layer hidden
    dropout=0.1,
)

# Input: X from GroupFeatureBuilder
# X shape: (M, 516) = (M, 256 + 256 + 1 + 3)
logits = switcher(X)  # (M,)
"""

import torch
import torch.nn as nn

from robot_nav.models.MARL.switcher import (
    GroupFeatureBuilder,
    GroupSwitcher,
    GroupSwitcherWithBaseline,
    pairwise_logistic_ranking_loss,
    hinge_ranking_loss,
    build_pairs_from_scores,
    build_pairs_from_ranking,
    compute_ranking_accuracy,
    compute_top1_accuracy,
)


def test_feature_builder():
    """Test GroupFeatureBuilder with various input configurations."""
    print("\n" + "="*60)
    print("Testing GroupFeatureBuilder")
    print("="*60)
    
    # Configuration
    n_robots = 6
    n_obs = 4
    embed_dim = 256
    
    # Define groups (matching workspace conventions)
    groups = [
        [0],           # Size 1
        [1],
        [0, 1],        # Size 2
        [2, 3],
        [0, 1, 2],     # Size 3
        [3, 4, 5],
    ]
    
    # Create fake embeddings
    h = torch.randn(n_robots, embed_dim)
    h_glob = torch.randn(embed_dim)
    
    # Create fake attention weights (matching AttentionObstacle output format)
    attn_rr = torch.rand(n_robots, n_robots)
    attn_rr = attn_rr / attn_rr.sum(dim=1, keepdim=True)  # Normalize
    attn_ro = torch.rand(n_robots, n_obs)
    attn_ro = attn_ro / attn_ro.sum(dim=1, keepdim=True)
    
    # Create extra features
    extra = {
        "dist_to_goal": torch.rand(n_robots) * 10,  # 0-10 meters
        "clearance": torch.rand(n_robots) * 2,       # 0-2 meters
    }
    
    # Test 1: Basic feature builder (no extras)
    print("\n1. Basic feature builder (no extras):")
    builder_basic = GroupFeatureBuilder(
        embed_dim=embed_dim,
        pooling="mean",
    )
    X_basic = builder_basic(h, groups)
    print(f"   Output shape: {X_basic.shape}")
    print(f"   Expected: ({len(groups)}, {builder_basic.output_dim})")
    assert X_basic.shape == (len(groups), builder_basic.output_dim)
    print("   ✓ Shape check passed")
    
    # Test 2: With attention weights
    print("\n2. With attention weights:")
    X_attn = builder_basic(h, groups, h_glob=h_glob, attn_rr=attn_rr, attn_ro=attn_ro)
    print(f"   Output shape: {X_attn.shape}")
    # Check that attention stats are not all zeros
    attn_stats_start = embed_dim + embed_dim + 1  # h_g + h_glob + size
    attn_stats = X_attn[:, attn_stats_start:attn_stats_start+3]
    print(f"   Attention stats sample (group 0): {attn_stats[0].tolist()}")
    assert not torch.allclose(attn_stats, torch.zeros_like(attn_stats))
    print("   ✓ Attention stats are non-zero")
    
    # Test 3: Without attention (should not crash, returns zeros)
    print("\n3. Without attention:")
    X_no_attn = builder_basic(h, groups, h_glob=h_glob)
    attn_stats_no = X_no_attn[:, attn_stats_start:attn_stats_start+3]
    print(f"   Attention stats (should be zeros): {attn_stats_no[0].tolist()}")
    assert torch.allclose(attn_stats_no, torch.zeros_like(attn_stats_no))
    print("   ✓ No crash, attention stats are zeros")
    
    # Test 4: With extra features
    print("\n4. With extra features:")
    builder_extra = GroupFeatureBuilder(
        embed_dim=embed_dim,
        pooling="mean",
        extra_feature_names=["dist_to_goal", "clearance"],
        extra_aggregations=["mean", "min"],
    )
    X_extra = builder_extra(h, groups, h_glob=h_glob, attn_rr=attn_rr, attn_ro=attn_ro, extra=extra)
    print(f"   Output shape: {X_extra.shape}")
    print(f"   Expected dim: {builder_extra.output_dim}")
    assert X_extra.shape == (len(groups), builder_extra.output_dim)
    print("   ✓ Extra features included")
    
    # Test 5: Missing extra features (should not crash)
    print("\n5. Missing extra features:")
    X_missing = builder_extra(h, groups)  # No extra dict provided
    assert X_missing.shape == (len(groups), builder_extra.output_dim)
    print("   ✓ No crash when extra features missing")
    
    # Test 6: Max pooling
    print("\n6. Max pooling:")
    builder_max = GroupFeatureBuilder(embed_dim=embed_dim, pooling="max")
    X_max = builder_max(h, groups)
    assert X_max.shape == (len(groups), builder_max.output_dim)
    print("   ✓ Max pooling works")
    
    # Test 7: Batched input
    print("\n7. Batched input:")
    h_batch = torch.randn(2, n_robots, embed_dim)
    attn_rr_batch = torch.rand(2, n_robots, n_robots)
    X_batch = builder_basic(h_batch, groups, attn_rr=attn_rr_batch)
    # Currently processes first batch element only
    assert X_batch.shape == (len(groups), builder_basic.output_dim)
    print("   ✓ Batched input handled")
    
    print("\n✓ All GroupFeatureBuilder tests passed!")
    return builder_extra.output_dim


def test_switcher_network(input_dim: int):
    """Test GroupSwitcher network."""
    print("\n" + "="*60)
    print("Testing GroupSwitcher Network")
    print("="*60)
    
    n_groups = 6
    
    # Create fake group features
    X = torch.randn(n_groups, input_dim)
    
    # Test 1: Basic forward pass
    print("\n1. Basic forward pass:")
    switcher = GroupSwitcher(input_dim=input_dim, hidden_dim=64, dropout=0.1)
    logits = switcher(X)
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {logits.shape}")
    assert logits.shape == (n_groups,)
    print("   ✓ Output shape correct")
    
    # Test 2: Group selection (argmax)
    print("\n2. Group selection (argmax):")
    selected = switcher.select_group(logits, mode="argmax")
    print(f"   Logits: {logits.tolist()}")
    print(f"   Selected group: {selected}")
    assert selected == logits.argmax().item()
    print("   ✓ Argmax selection correct")
    
    # Test 3: Group selection (softmax sampling)
    print("\n3. Group selection (softmax sampling):")
    selected_soft = switcher.select_group(logits, mode="softmax")
    print(f"   Selected group (softmax): {selected_soft}")
    assert 0 <= selected_soft < n_groups
    print("   ✓ Softmax selection in valid range")
    
    # Test 4: Sample with temperature
    print("\n4. Sample with temperature:")
    idx, log_prob = switcher.sample_group(logits, temperature=0.5)
    print(f"   Sampled index: {idx}, log_prob: {log_prob.item():.4f}")
    assert 0 <= idx < n_groups
    print("   ✓ Temperature sampling works")
    
    # Test 5: Get selection probabilities
    print("\n5. Selection probabilities:")
    probs = switcher.get_selection_probs(logits, temperature=1.0)
    print(f"   Probs: {probs.tolist()}")
    assert abs(probs.sum().item() - 1.0) < 1e-5
    print("   ✓ Probabilities sum to 1")
    
    # Test 6: Switcher with baseline
    print("\n6. Switcher with baseline:")
    switcher_baseline = GroupSwitcherWithBaseline(input_dim=input_dim, hidden_dim=64)
    logits_b, value = switcher_baseline(X)
    print(f"   Logits shape: {logits_b.shape}")
    print(f"   Value: {value.item():.4f}")
    assert logits_b.shape == (n_groups,)
    print("   ✓ Baseline version works")
    
    print("\n✓ All GroupSwitcher tests passed!")


def test_ranking_losses():
    """Test ranking loss functions."""
    print("\n" + "="*60)
    print("Testing Ranking Losses")
    print("="*60)
    
    n_groups = 5
    
    # Create fake logits and ground-truth scores
    logits = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9], requires_grad=True)
    gt_scores = torch.tensor([0.9, 0.3, 0.7, 0.1, 0.8])  # Ground truth
    
    # Test 1: Build pairs from scores
    print("\n1. Build pairs from scores:")
    pairs = build_pairs_from_scores(gt_scores, margin=0.0)
    print(f"   Number of pairs: {len(pairs)}")
    print(f"   Sample pairs: {pairs[:5].tolist()}")
    # Check pairs are valid (pos score > neg score)
    for i, j in pairs.tolist():
        assert gt_scores[i] > gt_scores[j], f"Invalid pair: {i}, {j}"
    print("   ✓ All pairs valid")
    
    # Test 2: Build pairs from ranking
    print("\n2. Build pairs from ranking:")
    ranking = [0, 4, 2, 1, 3]  # Best to worst
    pairs_rank = build_pairs_from_ranking(ranking)
    print(f"   Ranking: {ranking}")
    print(f"   Number of pairs: {len(pairs_rank)}")
    expected_n_pairs = len(ranking) * (len(ranking) - 1) // 2
    assert len(pairs_rank) == expected_n_pairs
    print("   ✓ Correct number of pairs")
    
    # Test 3: Pairwise logistic loss
    print("\n3. Pairwise logistic loss:")
    loss_log = pairwise_logistic_ranking_loss(logits, pairs)
    print(f"   Loss: {loss_log.item():.4f}")
    loss_log.backward()
    assert logits.grad is not None
    print(f"   Gradient norm: {logits.grad.norm().item():.4f}")
    print("   ✓ Backward pass works")
    
    # Reset gradient
    logits = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9], requires_grad=True)
    
    # Test 4: Hinge loss
    print("\n4. Hinge ranking loss:")
    loss_hinge = hinge_ranking_loss(logits, pairs, margin=0.5)
    print(f"   Loss (margin=0.5): {loss_hinge.item():.4f}")
    loss_hinge.backward()
    print("   ✓ Hinge loss works")
    
    # Test 5: Empty pairs (edge case)
    print("\n5. Empty pairs (edge case):")
    empty_pairs = torch.zeros(0, 2, dtype=torch.long)
    loss_empty = pairwise_logistic_ranking_loss(logits.detach(), empty_pairs)
    assert loss_empty.item() == 0.0
    print("   ✓ Handles empty pairs")
    
    # Test 6: Ranking accuracy
    print("\n6. Ranking accuracy:")
    # Perfect predictions
    perfect_logits = gt_scores.clone()
    acc_perfect = compute_ranking_accuracy(perfect_logits, pairs)
    print(f"   Accuracy with perfect logits: {acc_perfect:.2%}")
    assert acc_perfect == 1.0
    # Random predictions
    random_logits = torch.rand(n_groups)
    acc_random = compute_ranking_accuracy(random_logits, pairs)
    print(f"   Accuracy with random logits: {acc_random:.2%}")
    print("   ✓ Accuracy computation works")
    
    # Test 7: Top-1 accuracy
    print("\n7. Top-1 accuracy:")
    best_group = gt_scores.argmax().item()
    is_correct = compute_top1_accuracy(perfect_logits, best_group)
    print(f"   Best group (GT): {best_group}")
    print(f"   Top-1 correct: {is_correct}")
    assert is_correct
    print("   ✓ Top-1 accuracy works")
    
    print("\n✓ All ranking loss tests passed!")


def test_end_to_end():
    """Test complete pipeline: features -> network -> loss."""
    print("\n" + "="*60)
    print("Testing End-to-End Pipeline")
    print("="*60)
    
    # Configuration
    n_robots = 6
    n_obs = 4
    embed_dim = 256
    
    # Define groups
    groups = [
        [0], [1],           # Size 1
        [0, 1], [2, 3],     # Size 2
        [0, 1, 2], [3, 4, 5]  # Size 3
    ]
    
    # Fake data
    h = torch.randn(n_robots, embed_dim)
    h_glob = h.mean(dim=0)
    attn_rr = torch.rand(n_robots, n_robots)
    attn_ro = torch.rand(n_robots, n_obs)
    extra = {
        "dist_to_goal": torch.rand(n_robots) * 10,
        "clearance": torch.rand(n_robots) * 2,
    }
    
    # Build features
    print("\n1. Building group features:")
    builder = GroupFeatureBuilder(
        embed_dim=embed_dim,
        extra_feature_names=["dist_to_goal", "clearance"],
        extra_aggregations=["mean", "min"],
    )
    X = builder(h, groups, h_glob=h_glob, attn_rr=attn_rr, attn_ro=attn_ro, extra=extra)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Create switcher
    print("\n2. Creating switcher network:")
    switcher = GroupSwitcher(
        input_dim=builder.output_dim,
        hidden_dim=64,
        dropout=0.1,
        num_layers=2,
    )
    print(f"   Network parameters: {sum(p.numel() for p in switcher.parameters())}")
    
    # Forward pass
    print("\n3. Forward pass:")
    logits = switcher(X)
    print(f"   Logits: {logits.detach().tolist()}")
    
    # Simulate ground-truth scores (e.g., from oracle)
    print("\n4. Computing ranking loss:")
    gt_scores = torch.tensor([0.5, 0.4, 0.8, 0.7, 0.9, 0.6])  # Group 4 is best
    pairs = build_pairs_from_scores(gt_scores)
    print(f"   Number of pairs: {len(pairs)}")
    
    loss = pairwise_logistic_ranking_loss(logits, pairs)
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    print("\n5. Backward pass:")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in switcher.parameters() if p.grad is not None)
    print(f"   Total gradient norm: {grad_norm:.4f}")
    
    # Select group
    print("\n6. Group selection:")
    selected = switcher.select_group(logits.detach())
    print(f"   Selected group index: {selected}")
    print(f"   Selected group: {groups[selected]}")
    
    # Accuracy
    print("\n7. Evaluation metrics:")
    acc = compute_ranking_accuracy(logits.detach(), pairs)
    top1 = compute_top1_accuracy(logits.detach(), gt_scores.argmax().item())
    print(f"   Pairwise accuracy: {acc:.2%}")
    print(f"   Top-1 correct: {top1}")
    
    print("\n✓ End-to-end pipeline works!")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Group Switcher Module Demo")
    print("#"*60)
    
    # Run tests
    input_dim = test_feature_builder()
    test_switcher_network(input_dim)
    test_ranking_losses()
    test_end_to_end()
    
    print("\n" + "#"*60)
    print("# All tests passed! ✓")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
