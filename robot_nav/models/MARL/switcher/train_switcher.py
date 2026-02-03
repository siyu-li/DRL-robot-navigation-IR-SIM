"""
Training script for the Group Switcher module.

This script trains the GroupSwitcher network using data collected from oracle rollouts.
The oracle evaluates different group selections and provides quality scores for ranking.

Usage:
    python -m robot_nav.models.MARL.switcher.train_switcher

Data Format:
    See `OracleDataFormat` class and `collect_oracle_data.py` for details on
    how to collect and format training data.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.switcher import (
    GroupFeatureBuilder,
    GroupSwitcher,
    GroupSwitcherWithBaseline,
    pairwise_logistic_ranking_loss,
    hinge_ranking_loss,
    build_pairs_from_scores,
    compute_ranking_accuracy,
    compute_top1_accuracy,
)


# =============================================================================
# Configuration Dictionary - Edit these values directly
# =============================================================================
CONFIG = {
    # Data configuration
    "data_path": "robot_nav/models/MARL/switcher/data/oracle_data.pt",
    "embed_dim": 256,
    "extra_features": ["dist_to_goal", "clearance"],
    "extra_aggregations": ["mean", "min"],
    
    # Model architecture
    "embed_hidden": 256,            # Tower 1 output dimension
    "scalar_hidden": 32,            # Tower 2 output dimension
    "fusion_hidden": 256,           # Fusion layer hidden dimension
    "dropout": 0.1,
    "use_baseline": False,          # Use GroupSwitcherWithBaseline for RL training
    
    # Training configuration
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "loss_type": "logistic",        # "logistic" or "hinge"
    "hinge_margin": 1.0,            # Only used if loss_type="hinge"
    
    # Validation configuration
    "val_split": 0.1,               # Fraction of data for validation
    "eval_every": 5,                # Validate every N epochs
    
    # Saving configuration
    "save_dir": "robot_nav/models/MARL/switcher/runs/switcher",
    "save_every": 10,               # Save checkpoint every N epochs
    
    # Device configuration
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Resume training (set to checkpoint path to resume, or None)
    "resume": None,
}


# =============================================================================
# Data Format Specification
# =============================================================================

@dataclass
class OracleDataFormat:
    """
    Specification for oracle data collection format.
    
    When running the oracle (e.g., simulation rollouts, expert demonstrations),
    collect data in this format and save using torch.save().
    
    Required fields per sample:
    ---------------------------
    - h: Tensor[N, d] - Per-robot embeddings from GAT backbone
    - groups: List[List[int]] - Candidate groups (M groups)
    - group_scores: Tensor[M] - Oracle quality score for each group
        Higher score = better group. Can be:
        - Success rate from rollouts
        - Negative collision count
        - Progress toward goal
        - Expert preference score
    
    Optional fields:
    ----------------
    - h_glob: Tensor[d] - Global embedding (if not provided, computed as mean of h)
    - attn_rr: Tensor[N, N] - Robot-robot attention weights
    - attn_ro: Tensor[N, N_obs] - Robot-obstacle attention weights
    - extra: Dict[str, Tensor[N]] - Extra per-robot features
        - "dist_to_goal": Distance to goal for each robot
        - "clearance": Minimum obstacle clearance for each robot
    - metadata: Dict - Any additional info (scenario_id, timestamp, etc.)
    
    Example data structure:
    -----------------------
    data = {
        "samples": [
            {
                "h": torch.randn(6, 256),           # 6 robots, 256-dim embeddings
                "groups": [[0], [1], [0,1], [2,3]], # 4 candidate groups
                "group_scores": torch.tensor([0.3, 0.4, 0.9, 0.7]),  # Oracle scores
                "attn_rr": torch.rand(6, 6),        # Optional
                "attn_ro": torch.rand(6, 4),        # Optional (4 obstacles)
                "extra": {
                    "dist_to_goal": torch.tensor([2.1, 3.5, 1.2, 4.0, 2.8, 3.2]),
                    "clearance": torch.tensor([0.5, 0.8, 0.3, 0.6, 0.4, 0.7]),
                },
            },
            # ... more samples
        ],
        "config": {
            "embed_dim": 256,
            "n_robots": 6,
            "collection_method": "simulation_rollout",  # or "expert_demo"
            "timestamp": "2026-02-02",
        }
    }
    torch.save(data, "oracle_data.pt")
    """
    pass


# =============================================================================
# Dataset
# =============================================================================

class SwitcherDataset(Dataset):
    """
    Dataset for training the GroupSwitcher.
    
    Each sample contains:
    - Group features (built from embeddings + attention + extras)
    - Oracle scores for ranking supervision
    """
    
    def __init__(
        self,
        data_path: str,
        embed_dim: int = 256,
        extra_feature_names: Optional[List[str]] = None,
        extra_aggregations: Optional[List[str]] = None,
    ):
        """
        Args:
            data_path: Path to oracle data file (.pt)
            embed_dim: Dimension of robot embeddings
            extra_feature_names: Names of extra features to include
            extra_aggregations: Aggregation methods for extra features
        """
        self.data = torch.load(data_path)
        self.samples = self.data["samples"]
        self.config = self.data.get("config", {})
        
        # Feature builder
        self.feature_builder = GroupFeatureBuilder(
            embed_dim=embed_dim,
            extra_feature_names=extra_feature_names or [],
            extra_aggregations=extra_aggregations,
        )
        
        # Validate and preprocess
        self._validate_data()
    
    def _validate_data(self):
        """Validate data format."""
        assert len(self.samples) > 0, "No samples in dataset"
        
        sample = self.samples[0]
        assert "h" in sample, "Missing 'h' (robot embeddings)"
        assert "groups" in sample, "Missing 'groups'"
        assert "group_scores" in sample, "Missing 'group_scores'"
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"  Embedding dim: {sample['h'].shape[-1]}")
        print(f"  Groups per sample: {len(sample['groups'])}")
        if "extra" in sample:
            print(f"  Extra features: {list(sample['extra'].keys())}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Extract data
        h = sample["h"]
        groups = sample["groups"]
        group_scores = sample["group_scores"]
        
        # Optional fields
        h_glob = sample.get("h_glob", None)
        attn_rr = sample.get("attn_rr", None)
        attn_ro = sample.get("attn_ro", None)
        extra = sample.get("extra", None)
        
        # Build group features
        X = self.feature_builder(
            h=h,
            groups=groups,
            h_glob=h_glob,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            extra=extra,
        )
        
        # Build ranking pairs
        pairs = build_pairs_from_scores(group_scores)
        
        # Best group index (for top-1 accuracy)
        best_idx = group_scores.argmax().item()
        
        return {
            "X": X,                      # (M, D) group features
            "group_scores": group_scores, # (M,) oracle scores
            "pairs": pairs,              # (K, 2) ranking pairs
            "best_idx": best_idx,        # int: best group index
            "n_groups": len(groups),     # int: number of groups
        }
    
    @property
    def feature_dim(self) -> int:
        """Output dimension of group features."""
        return self.feature_builder.output_dim
    
    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self.feature_builder.embed_dim
    
    @property
    def scalar_dim(self) -> int:
        """Scalar feature dimension (size + attn + extras)."""
        return 1 + 3 + len(self.feature_builder.extra_feature_names)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-size groups.
    
    Since different samples may have different numbers of groups,
    we process them individually during training.
    """
    # For now, just return the batch as a list
    # Each sample is processed individually in the training loop
    return batch


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_path: str = "oracle_data.pt"
    embed_dim: int = 256
    extra_features: List[str] = field(default_factory=lambda: ["dist_to_goal", "clearance"])
    extra_aggregations: List[str] = field(default_factory=lambda: ["mean", "min"])
    
    # Model
    embed_hidden: int = 256
    scalar_hidden: int = 32
    fusion_hidden: int = 256
    dropout: float = 0.1
    use_baseline: bool = False
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss_type: str = "logistic"  # "logistic" or "hinge"
    hinge_margin: float = 1.0
    
    # Validation
    val_split: float = 0.1
    eval_every: int = 5
    
    # Saving
    save_dir: str = "runs/switcher"
    save_every: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SwitcherTrainer:
    """Trainer for GroupSwitcher."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup directories
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(config), f, indent=2)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.save_dir))
        
        # Load data
        self._setup_data()
        
        # Create model
        self._setup_model()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.01,
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
    
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        config = self.config
        
        # Load full dataset
        full_dataset = SwitcherDataset(
            data_path=config.data_path,
            embed_dim=config.embed_dim,
            extra_feature_names=config.extra_features,
            extra_aggregations=config.extra_aggregations,
        )
        
        # Split into train/val
        n_val = int(len(full_dataset) * config.val_split)
        n_train = len(full_dataset) - n_val
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [n_train, n_val]
        )
        
        print(f"Train samples: {n_train}, Val samples: {n_val}")
        
        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        # Store dimensions
        self.embed_dim = full_dataset.embed_dim
        self.scalar_dim = full_dataset.scalar_dim
    
    def _setup_model(self):
        """Setup model."""
        config = self.config
        
        if config.use_baseline:
            self.model = GroupSwitcherWithBaseline(
                embed_dim=self.embed_dim,
                scalar_dim=self.scalar_dim,
                embed_hidden=config.embed_hidden,
                scalar_hidden=config.scalar_hidden,
                fusion_hidden=config.fusion_hidden,
                dropout=config.dropout,
            )
        else:
            self.model = GroupSwitcher(
                embed_dim=self.embed_dim,
                scalar_dim=self.scalar_dim,
                embed_hidden=config.embed_hidden,
                scalar_hidden=config.scalar_hidden,
                fusion_hidden=config.fusion_hidden,
                dropout=config.dropout,
            )
        
        self.model = self.model.to(self.device)
        
        # Print model info
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranking loss."""
        if self.config.loss_type == "logistic":
            return pairwise_logistic_ranking_loss(logits, pairs)
        elif self.config.loss_type == "hinge":
            return hinge_ranking_loss(logits, pairs, margin=self.config.hinge_margin)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        total_top1 = 0.0
        n_samples = 0
        
        for batch in self.train_loader:
            # Process each sample in batch individually
            batch_loss = 0.0
            batch_acc = 0.0
            batch_top1 = 0.0
            
            for sample in batch:
                X = sample["X"].to(self.device)
                pairs = sample["pairs"].to(self.device)
                best_idx = sample["best_idx"]
                
                # Forward
                if self.config.use_baseline:
                    logits, value = self.model(X)
                else:
                    logits = self.model(X)
                
                # Loss
                loss = self.compute_loss(logits, pairs)
                batch_loss += loss
                
                # Metrics
                with torch.no_grad():
                    acc = compute_ranking_accuracy(logits, pairs)
                    top1 = float(compute_top1_accuracy(logits, best_idx))
                    batch_acc += acc
                    batch_top1 += top1
            
            # Average over batch
            batch_size = len(batch)
            batch_loss = batch_loss / batch_size
            
            # Backward
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += batch_loss.item() * batch_size
            total_acc += batch_acc
            total_top1 += batch_top1
            n_samples += batch_size
            
            self.global_step += 1
        
        # Epoch metrics
        metrics = {
            "train/loss": total_loss / n_samples,
            "train/pair_acc": total_acc / n_samples,
            "train/top1_acc": total_top1 / n_samples,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        total_top1 = 0.0
        n_samples = 0
        
        for batch in self.val_loader:
            for sample in batch:
                X = sample["X"].to(self.device)
                pairs = sample["pairs"].to(self.device)
                best_idx = sample["best_idx"]
                
                # Forward
                if self.config.use_baseline:
                    logits, value = self.model(X)
                else:
                    logits = self.model(X)
                
                # Loss
                loss = self.compute_loss(logits, pairs)
                
                # Metrics
                acc = compute_ranking_accuracy(logits, pairs)
                top1 = float(compute_top1_accuracy(logits, best_idx))
                
                total_loss += loss.item()
                total_acc += acc
                total_top1 += top1
                n_samples += 1
        
        metrics = {
            "val/loss": total_loss / n_samples,
            "val/pair_acc": total_acc / n_samples,
            "val/top1_acc": total_top1 / n_samples,
        }
        
        return metrics
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": vars(self.config),
        }
        
        path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_acc = checkpoint["best_val_acc"]
        
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}\n")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch + 1
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log training metrics
            for k, v in train_metrics.items():
                self.writer.add_scalar(k, v, self.epoch)
            
            # Validate
            if self.epoch % self.config.eval_every == 0:
                val_metrics = self.validate()
                
                for k, v in val_metrics.items():
                    self.writer.add_scalar(k, v, self.epoch)
                
                # Check for best model
                if val_metrics["val/top1_acc"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["val/top1_acc"]
                    self.save_checkpoint("best")
            else:
                val_metrics = {}
            
            # Learning rate
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar("train/lr", lr, self.epoch)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {self.epoch:3d}/{self.config.epochs} | "
                  f"Loss: {train_metrics['train/loss']:.4f} | "
                  f"PairAcc: {train_metrics['train/pair_acc']:.1%} | "
                  f"Top1: {train_metrics['train/top1_acc']:.1%} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.1f}s", end="")
            
            if val_metrics:
                print(f" | ValLoss: {val_metrics['val/loss']:.4f} | "
                      f"ValTop1: {val_metrics['val/top1_acc']:.1%}", end="")
            print()
            
            # Save periodic checkpoint
            if self.epoch % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{self.epoch}")
        
        # Final save
        self.save_checkpoint("final")
        self.writer.close()
        
        print(f"\nTraining complete!")
        print(f"Best validation Top-1 accuracy: {self.best_val_acc:.1%}")
        print(f"Checkpoints saved to: {self.save_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training function."""
    cfg = CONFIG
    
    print("=" * 60)
    print("Group Switcher Training")
    print("=" * 60)
    print(f"Data path: {cfg['data_path']}")
    print(f"Save directory: {cfg['save_dir']}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Learning rate: {cfg['lr']}")
    print(f"Loss type: {cfg['loss_type']}")
    print(f"Device: {cfg['device']}")
    print("=" * 60 + "\n")
    
    # Create config dataclass from dictionary
    config = TrainingConfig(
        data_path=cfg["data_path"],
        embed_dim=cfg["embed_dim"],
        extra_features=cfg["extra_features"],
        extra_aggregations=cfg["extra_aggregations"],
        embed_hidden=cfg["embed_hidden"],
        scalar_hidden=cfg["scalar_hidden"],
        fusion_hidden=cfg["fusion_hidden"],
        dropout=cfg["dropout"],
        use_baseline=cfg["use_baseline"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        loss_type=cfg["loss_type"],
        hinge_margin=cfg["hinge_margin"],
        val_split=cfg["val_split"],
        eval_every=cfg["eval_every"],
        save_dir=cfg["save_dir"],
        save_every=cfg["save_every"],
        device=cfg["device"],
    )
    
    # Create trainer
    trainer = SwitcherTrainer(config)
    
    # Resume if specified
    if cfg["resume"]:
        trainer.load_checkpoint(cfg["resume"])
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
