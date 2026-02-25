"""
Training script for the Group Switcher module.

This script trains the GroupSwitcher network using data collected from oracle rollouts.
The oracle evaluates different group selections and provides quality scores for ranking.

Usage:
    python -m robot_nav.scripts.train_switcher

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

from robot_nav.models.MARL.switcher.supervised import (
    GroupFeatureBuilder,
    DEFAULT_EXTRA_GROUP,
    DEFAULT_EXTRA_GLOBAL,
    GroupSwitcher,
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
    "data_path": "robot_nav/models/MARL/switcher/data/oracle_data_14robots_decouple_couple_group_len1200_success.pt",
    "embed_dim": 512,              # Dimension of per-robot embeddings: Will be adjust based on data if None
    
    # GroupFeatureBuilder config
    #   Features 1-5 are always on (size_feat, coupling_mode, A_in, A_out, A_obs).
    #   extra_group: list of (extra_key, aggregation) for per-group scalars
    #                aggregated over group members. Set to [] to disable.
    #   extra_global: list of extra_key names for global scalars (same for
    #                 every group). Set to [] to disable.
    #   scalar_dim = 5 + len(extra_group) + len(extra_global) + 1 (urgency_flag)
    "max_group_size": 7,
    "rotation_coupling_threshold": 3,
    "extra_group": [
        ("dist_to_goal", "mean"),   # mean_dist_goal_g
        ("dist_to_goal", "min"),    # min_dist_goal_g
        ("clearance",    "min"),    # min_clearance_g
        ("reached",      "mean"),   # frac_reached_g
        ("heading_error","mean"),   # mean_heading_err_g
    ],
    "extra_global": [
        "var_dist_goal_global",     # distance variance (sync signal)
        "frac_reached_global",      # global completion fraction
        "steps_elapsed_frac",       # time pressure
    ],
    
    # Urgency flag: Binary indicator for single-robot groups with urgent robots
    #   1.0 if group size == 1 AND that robot is urgent
    #   0.0 otherwise (all multi-robot groups OR non-urgent single robots)
    "use_urgency_flag": False,
    
    # Model architecture
    "embed_hidden": 256,            # Tower 1 output dimension
    "scalar_hidden": 64,            # Tower 2 output dimension
    "fusion_hidden": 256,           # Fusion layer hidden dimension
    "dropout": 0.1,
    
    # Training configuration
    "epochs": 100,
    "batch_size": 64,
    "lr": 3e-4,
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
                "h": torch.randn(6, 512),           # 6 robots, 512-dim embeddings
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
            "embed_dim": 512,
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
    - Urgency flag (binary scalar appended to each group's features)
    """
    
    def __init__(
        self,
        data_path: str,
        embed_dim: Optional[int] = None,
        max_group_size: int = 7,
        rotation_coupling_threshold: int = 3,
        extra_group: Optional[List[Tuple[str, str]]] = None,
        extra_global: Optional[List[str]] = None,
        use_urgency_flag: bool = True,
    ):
        """
        Args:
            data_path: Path to oracle data file (.pt)
            embed_dim: Dimension of robot embeddings. If None, inferred from data.
            max_group_size: Normalisation constant for size_feat.
            rotation_coupling_threshold: Groups > this size get coupling_mode=1.
            extra_group: Per-group extra features as ``(key, agg)`` pairs.
                ``None`` → defaults.  ``[]`` → disabled.
            extra_global: Global extra feature key names.
                ``None`` → defaults.  ``[]`` → disabled.
            use_urgency_flag: If True, append urgency flag to group features.
        """
        self.data = torch.load(data_path)
        self.samples = self.data["samples"]
        self.config = self.data.get("config", {})
        self.use_urgency_flag = use_urgency_flag
        
        # Infer embed_dim from data if not provided
        if embed_dim is None:
            embed_dim = self.samples[0]["h"].shape[-1]
        
        # Feature builder (configurable scalar layout)
        self.feature_builder = GroupFeatureBuilder(
            embed_dim=embed_dim,
            max_group_size=max_group_size,
            rotation_coupling_threshold=rotation_coupling_threshold,
            extra_group=extra_group,
            extra_global=extra_global,
        )
        
        # Validate and preprocess
        self._validate_data()
        
        # Pre-compute all features once so __getitem__ is O(1)
        print("Pre-computing features for all samples (runs once)...")
        self._precomputed = self._precompute_all()
        print("Pre-computation done.")
    
    def _precompute_all(self) -> List[Dict[str, torch.Tensor]]:
        """Build and cache features + pairs for every sample up front."""
        cache = []
        for sample in self.samples:
            h            = sample["h"]
            groups       = sample["groups"]
            group_scores = sample["group_scores"]
            h_glob  = sample.get("h_glob",  None)
            attn_rr = sample.get("attn_rr", None)
            attn_ro = sample.get("attn_ro", None)
            extra   = sample.get("extra",   None)

            X = self.feature_builder(
                h=h, groups=groups, h_glob=h_glob,
                attn_rr=attn_rr, attn_ro=attn_ro, extra=extra,
            )  # (M, D_base)

            if self.use_urgency_flag:
                urgency_flags = self._compute_urgency_flags(groups, extra)
                X = torch.cat([X, urgency_flags.unsqueeze(1)], dim=1)

            pairs    = build_pairs_from_scores(group_scores)
            best_idx = int(group_scores.argmax().item())

            cache.append({
                "X":           X,
                "group_scores": group_scores,
                "pairs":        pairs,
                "best_idx":     best_idx,
                "n_groups":     len(groups),
            })
        return cache
    
    def _validate_data(self):
        """Validate data format."""
        assert len(self.samples) > 0, "No samples in dataset"
        
        sample = self.samples[0]
        assert "h" in sample, "Missing 'h' (robot embeddings)"
        assert "groups" in sample, "Missing 'groups'"
        assert "group_scores" in sample, "Missing 'group_scores'"
        
        fb = self.feature_builder
        print(f"Loaded {len(self.samples)} samples")
        print(f"  Embedding dim: {sample['h'].shape[-1]}")
        print(f"  Groups per sample: {len(sample['groups'])}")
        base_scalar_dim = fb.scalar_dim
        urgency_dim = 1 if self.use_urgency_flag else 0
        total_scalar_dim = base_scalar_dim + urgency_dim
        print(f"  Scalar dim: {total_scalar_dim}  "
              f"(5 base + {len(fb.extra_group)} group + {len(fb.extra_global)} global"
              f"{' + 1 urgency_flag' if self.use_urgency_flag else ''})")
        if fb.extra_group:
            print(f"  extra_group: {fb.extra_group}")
        if fb.extra_global:
            print(f"  extra_global: {fb.extra_global}")
        if self.use_urgency_flag:
            print(f"  urgency_flag: binary (1.0 only for size-1 urgent groups)")
        if "extra" in sample:
            print(f"  Extra keys in data: {list(sample['extra'].keys())}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._precomputed[idx]

    def _getitem_slow(self, idx: int) -> Dict[str, torch.Tensor]:
        """Original (uncached) item getter — kept for reference."""
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
        
        # Build group features (without urgency flag)
        X = self.feature_builder(
            h=h,
            groups=groups,
            h_glob=h_glob,
            attn_rr=attn_rr,
            attn_ro=attn_ro,
            extra=extra,
        )  # (M, D_base)
        
        # Add urgency flag as additional scalar feature
        if self.use_urgency_flag:
            urgency_flags = self._compute_urgency_flags(groups, extra)  # (M,)
            X = torch.cat([X, urgency_flags.unsqueeze(1)], dim=1)  # (M, D_base + 1)
        
        # Build ranking pairs
        pairs = build_pairs_from_scores(group_scores)
        
        # Best group index (for top-1 accuracy)
        best_idx = group_scores.argmax().item()
        
        return {
            "X": X,                      # (M, D) group features (with urgency if enabled)
            "group_scores": group_scores, # (M,) oracle scores
            "pairs": pairs,              # (K, 2) ranking pairs
            "best_idx": best_idx,        # int: best group index
            "n_groups": len(groups),     # int: number of groups
        }
    
    def _compute_urgency_flags(
        self,
        groups: List[List[int]],
        extra: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute urgency flags for all groups.
        
        Urgency flag logic:
        - 1.0 if group size == 1 AND that robot is urgent (extra["urgency"][robot_id] == 1.0)
        - 0.0 otherwise (all multi-robot groups OR non-urgent single robots)
        
        Args:
            groups: List of M groups
            extra: Extra features dict containing "urgency" tensor of shape (N,)
            
        Returns:
            urgency_flags: Tensor of shape (M,) with values 0.0 or 1.0
        """
        if extra is None or "urgency" not in extra:
            # No urgency data available, all flags are 0
            return torch.zeros(len(groups), dtype=torch.float32)
        
        urgency_per_robot = extra["urgency"]  # (N,) with values 0.0 or 1.0
        
        urgency_flags = []
        for group in groups:
            if len(group) == 1:
                # Single-robot group: check if that robot is urgent
                robot_id = group[0]
                is_urgent = urgency_per_robot[robot_id].item()
                urgency_flags.append(is_urgent)
            else:
                # Multi-robot group: urgency flag is always 0
                urgency_flags.append(0.0)
        
        return torch.tensor(urgency_flags, dtype=torch.float32)
    
    @property
    def feature_dim(self) -> int:
        """Output dimension of group features (including urgency flag if enabled)."""
        base_dim = self.feature_builder.output_dim
        urgency_dim = 1 if self.use_urgency_flag else 0
        return base_dim + urgency_dim
    
    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self.feature_builder.embed_dim
    
    @property
    def scalar_dim(self) -> int:
        """Scalar feature dimension (5 base + extra_group + extra_global)."""
        return self.feature_builder.scalar_dim


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
    embed_dim: int = 512
    max_group_size: int = 7
    rotation_coupling_threshold: int = 3
    extra_group: Optional[List[Tuple[str, str]]] = None   # None → defaults
    extra_global: Optional[List[str]] = None               # None → defaults
    use_urgency_flag: bool = True  # Append urgency flag as additional scalar
    
    # Model
    embed_hidden: int = 256
    scalar_hidden: int = 32
    fusion_hidden: int = 256
    dropout: float = 0.1    
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
        
        # Load full dataset (embed_dim is inferred from data)
        full_dataset = SwitcherDataset(
            data_path=config.data_path,
            embed_dim=None,  # Infer from data
            max_group_size=config.max_group_size,
            rotation_coupling_threshold=config.rotation_coupling_threshold,
            extra_group=config.extra_group,
            extra_global=config.extra_global,
            use_urgency_flag=config.use_urgency_flag,
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
        
        # Store dimensions (scalar_dim includes urgency flag if enabled)
        self.embed_dim = full_dataset.embed_dim
        base_scalar_dim = full_dataset.feature_builder.scalar_dim
        urgency_dim = 1 if config.use_urgency_flag else 0
        self.scalar_dim = base_scalar_dim + urgency_dim
    
    def _setup_model(self):
        """Setup model."""
        config = self.config
        
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
            batch_size = len(batch)
            n_groups_list = [s["n_groups"] for s in batch]

            # ── Batched forward pass (when all samples share the same M) ──
            if len(set(n_groups_list)) == 1:
                X_batch = torch.stack([s["X"] for s in batch]).to(self.device)  # (B, M, D)
                B, M, D = X_batch.shape
                logits_batch = self.model(X_batch.view(B * M, D)).view(B, M)   # (B, M)
            else:
                logits_batch = None  # fall back to per-sample forward

            # ── Per-sample loss & metrics ──
            losses = []
            batch_acc = 0.0
            batch_top1 = 0.0

            for i, sample in enumerate(batch):
                if logits_batch is not None:
                    logits = logits_batch[i]
                else:
                    X = sample["X"].to(self.device)
                    logits = self.model(X)

                pairs    = sample["pairs"].to(self.device)
                best_idx = sample["best_idx"]

                losses.append(self.compute_loss(logits, pairs))

                with torch.no_grad():
                    batch_acc  += compute_ranking_accuracy(logits, pairs)
                    batch_top1 += float(compute_top1_accuracy(logits, best_idx))

            # Average loss over batch, then backward
            batch_loss = sum(losses) / batch_size

            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss  += batch_loss.item() * batch_size
            total_acc   += batch_acc
            total_top1  += batch_top1
            n_samples   += batch_size
            self.global_step += 1
        
        return {
            "train/loss":     total_loss  / n_samples,
            "train/pair_acc": total_acc   / n_samples,
            "train/top1_acc": total_top1  / n_samples,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        total_top1 = 0.0
        n_samples = 0
        
        for batch in self.val_loader:
            batch_size = len(batch)
            n_groups_list = [s["n_groups"] for s in batch]

            # ── Batched forward pass ──
            if len(set(n_groups_list)) == 1:
                X_batch = torch.stack([s["X"] for s in batch]).to(self.device)
                B, M, D = X_batch.shape
                logits_batch = self.model(X_batch.view(B * M, D)).view(B, M)
            else:
                logits_batch = None

            for i, sample in enumerate(batch):
                if logits_batch is not None:
                    logits = logits_batch[i]
                else:
                    X = sample["X"].to(self.device)
                    logits = self.model(X)

                pairs    = sample["pairs"].to(self.device)
                best_idx = sample["best_idx"]

                total_loss  += self.compute_loss(logits, pairs).item()
                total_acc   += compute_ranking_accuracy(logits, pairs)
                total_top1  += float(compute_top1_accuracy(logits, best_idx))
                n_samples   += 1
        
        return {
            "val/loss":     total_loss  / n_samples,
            "val/pair_acc": total_acc   / n_samples,
            "val/top1_acc": total_top1  / n_samples,
        }
    
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
        max_group_size=cfg["max_group_size"],
        rotation_coupling_threshold=cfg["rotation_coupling_threshold"],
        extra_group=cfg["extra_group"],
        extra_global=cfg["extra_global"],
        use_urgency_flag=cfg["use_urgency_flag"],
        embed_hidden=cfg["embed_hidden"],
        scalar_hidden=cfg["scalar_hidden"],
        fusion_hidden=cfg["fusion_hidden"],
        dropout=cfg["dropout"],
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
