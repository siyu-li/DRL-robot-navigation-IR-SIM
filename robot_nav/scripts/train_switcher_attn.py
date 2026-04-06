"""
Training script for the GroupSwitcher with attention-based group pooling.

Uses AttentionGroupPooling + AttnGroupFeatureBuilder instead of mean-pool +
hand-crafted scalars.  No scalar tower — group embeddings are fully learned.

Data format (minimal, per sample):
    {
        "h":            Tensor[N, embed_dim],   # per-robot GAT embeddings
        "groups":       List[List[int]],         # M candidate groups
        "group_scores": Tensor[M],               # oracle quality scores (higher = better)
        "h_glob":       Tensor[embed_dim],       # optional; computed as h.mean(dim=0) if absent
    }

Usage:
    python -m robot_nav.scripts.train_switcher_attn
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.switcher.supervised import (
    GroupSwitcher,
    pairwise_logistic_ranking_loss,
    hinge_ranking_loss,
    build_pairs_from_scores,
    compute_ranking_accuracy,
    compute_top1_accuracy,
)
from robot_nav.models.MARL.switcher.supervised.attn_feature_builder import AttnGroupFeatureBuilder
from robot_nav.models.MARL.switcher.config_loader import load_switcher_config, build_attn_pool


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data
    "data_path":    "robot_nav/models/MARL/switcher/data/oracle_data_14robots_decouple_couple_group_len1200_avev_success.pt",
    "embed_dim":    512,   # inferred from data if None

    # Switcher YAML (must have pooling: "attention")
    "switcher_config_path": "robot_nav/models/MARL/switcher/switcher_config.yaml",

    # Model architecture
    "embed_hidden":  256,
    "fusion_hidden": 256,
    "dropout":       0.1,

    # Training
    "epochs":       100,
    "batch_size":   64,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "loss_type":    "logistic",   # "logistic" or "hinge"
    "hinge_margin": 1.0,

    # Validation
    "val_split":  0.1,
    "eval_every": 5,

    # Checkpointing
    "save_dir":   "robot_nav/models/MARL/switcher/runs/switcher_attn",
    "save_every": 10,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Resume from checkpoint path, or None
    "resume": None,
}


# =============================================================================
# Dataset
# =============================================================================

class SwitcherAttnDataset(Dataset):
    """
    Dataset for attention-pooling switcher training.

    Each sample stores raw robot embeddings (h) and group indices so that the
    attention pooling module can be called *during* training.  This avoids
    pre-computing a multi-GB feature cache (which OOM-kills the process) and
    ensures that gradients flow correctly through attn_pool on every step.

    __getitem__ returns: h, groups, pairs, best_idx, n_groups.
    Feature computation (h → X) is done in the trainer's forward pass.
    """

    def __init__(
        self,
        data_path: str,
        embed_dim: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to oracle .pt file.
            embed_dim: Expected embedding dim; inferred from data if None.
        """
        data = torch.load(data_path, weights_only=False)
        samples = data["samples"]
        self.config = data.get("config", {})

        if embed_dim is None:
            embed_dim = samples[0]["h"].shape[-1]
        self.embed_dim = embed_dim

        self._validate(samples)
        print("Indexing dataset (pairs + best_idx)...")
        self._cache = self._build_index(samples)
        print(f"Dataset ready. ({len(self._cache)} samples)")

    # ------------------------------------------------------------------

    def _validate(self, samples):
        assert len(samples) > 0, "No samples in dataset."
        s = samples[0]
        for key in ("h", "groups", "group_scores"):
            assert key in s, f"Missing required key '{key}' in sample."
        print(f"Loaded {len(samples)} samples | embed_dim={s['h'].shape[-1]} "
              f"| groups/sample={len(s['groups'])}")

    def _build_index(self, samples) -> List[Dict]:
        """Pre-compute only cheap, weight-independent quantities."""
        cache = []
        for s in samples:
            group_scores = s["group_scores"]
            pairs    = build_pairs_from_scores(group_scores)
            best_idx = int(group_scores.argmax().item())
            cache.append({
                "h":       s["h"],          # (N, D) — raw embeddings, on CPU
                "groups":  s["groups"],     # List[List[int]]
                "pairs":   pairs,           # (K, 2)
                "best_idx": best_idx,       # int
                "n_groups": len(s["groups"]),
            })
        return cache

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx: int) -> Dict:
        return self._cache[idx]


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """Return batch as a list; samples have variable numbers of groups."""
    return batch


# =============================================================================
# Training config
# =============================================================================

@dataclass
class TrainingConfig:
    data_path:            str   = "oracle_data.pt"
    embed_dim:            int   = 512
    switcher_config_path: str   = "robot_nav/models/MARL/switcher/switcher_config.yaml"
    embed_hidden:         int   = 256
    fusion_hidden:        int   = 256
    dropout:              float = 0.1
    epochs:               int   = 100
    batch_size:           int   = 64
    lr:                   float = 3e-4
    weight_decay:         float = 1e-4
    loss_type:            str   = "logistic"
    hinge_margin:         float = 1.0
    val_split:            float = 0.1
    eval_every:           int   = 5
    save_dir:             str   = "runs/switcher_attn"
    save_every:           int   = 10
    device:               str   = "cpu"


# =============================================================================
# Trainer
# =============================================================================

class SwitcherAttnTrainer:
    """Trainer for GroupSwitcher + AttentionGroupPooling."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(config), f, indent=2)

        self.writer = SummaryWriter(log_dir=str(self.save_dir))

        self._setup_data()
        self._setup_model()

        # Single optimizer covers GroupSwitcher + AttentionGroupPooling
        all_params = list(self.model.parameters()) + list(self.attn_pool.parameters())
        self.optimizer = optim.AdamW(all_params, lr=config.lr, weight_decay=config.weight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )

        self.epoch       = 0
        self.global_step = 0
        self.best_val_acc = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_data(self):
        config = self.config

        sw_cfg = load_switcher_config(config.switcher_config_path)
        self._sw_cfg = sw_cfg

        if sw_cfg.pooling != "attention":
            raise ValueError(
                f"switcher_config has pooling='{sw_cfg.pooling}'. "
                "This trainer requires pooling='attention'."
            )

        embed_dim = config.embed_dim or 512
        self.attn_pool      = build_attn_pool(sw_cfg, embed_dim=embed_dim)
        self.feature_builder = AttnGroupFeatureBuilder(self.attn_pool, embed_dim=embed_dim)

        print(f"AttentionGroupPooling: n_heads={sw_cfg.attn_n_heads}, "
              f"score_hidden={sw_cfg.attn_score_hidden}")
        print(f"Group feature dim: {self.feature_builder.output_dim}  "
              f"(= 2 × embed_dim = 2 × {embed_dim})")

        full_dataset = SwitcherAttnDataset(
            data_path=config.data_path,
            embed_dim=embed_dim,
        )

        n_val   = int(len(full_dataset) * config.val_split)
        n_train = len(full_dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(full_dataset, [n_train, n_val])
        print(f"Train: {n_train}  |  Val: {n_val}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0,
        )

        self.embed_dim = embed_dim

    def _setup_model(self):
        config = self.config

        # scalar_dim=0: no scalar tower; fusion input = embed_hidden only
        self.model = GroupSwitcher(
            embed_dim=self.embed_dim,
            scalar_dim=0,
            embed_hidden=config.embed_hidden,
            fusion_hidden=config.fusion_hidden,
            dropout=config.dropout,
        ).to(self.device)

        self.attn_pool = self.attn_pool.to(self.device)

        n_model = sum(p.numel() for p in self.model.parameters())
        n_pool  = sum(p.numel() for p in self.attn_pool.parameters())
        print(f"GroupSwitcher params:     {n_model:,}")
        print(f"AttentionGroupPooling params: {n_pool:,}")
        print(f"Total trainable params:   {n_model + n_pool:,}")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, logits: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "logistic":
            return pairwise_logistic_ranking_loss(logits, pairs)
        elif self.config.loss_type == "hinge":
            return hinge_ranking_loss(logits, pairs, margin=self.config.hinge_margin)
        raise ValueError(f"Unknown loss_type: '{self.config.loss_type}'")

    # ------------------------------------------------------------------
    # Train / Validate
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.attn_pool.train()

        total_loss = total_acc = total_top1 = 0.0
        n_samples = 0

        for batch in self.train_loader:
            n_groups_list = [s["n_groups"] for s in batch]

            # Batched forward when all samples have the same number of groups
            if len(set(n_groups_list)) == 1:
                X_list = [
                    self.feature_builder(h=s["h"].to(self.device), groups=s["groups"])
                    for s in batch
                ]
                X_batch = torch.stack(X_list)         # (B, M, 2D)
                B, M, D = X_batch.shape
                logits_batch = self.model(X_batch.view(B * M, D)).view(B, M)
            else:
                logits_batch = None

            losses = []
            batch_acc = batch_top1 = 0.0

            for i, sample in enumerate(batch):
                if logits_batch is not None:
                    logits = logits_batch[i]
                else:
                    X = self.feature_builder(
                        h=sample["h"].to(self.device), groups=sample["groups"]
                    )
                    logits = self.model(X)
                pairs    = sample["pairs"].to(self.device)
                best_idx = sample["best_idx"]

                losses.append(self._compute_loss(logits, pairs))
                with torch.no_grad():
                    batch_acc  += compute_ranking_accuracy(logits, pairs)
                    batch_top1 += float(compute_top1_accuracy(logits, best_idx))

            batch_loss = sum(losses) / len(batch)
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.attn_pool.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            n = len(batch)
            total_loss  += batch_loss.item() * n
            total_acc   += batch_acc
            total_top1  += batch_top1
            n_samples   += n
            self.global_step += 1

        return {
            "train/loss":     total_loss  / n_samples,
            "train/pair_acc": total_acc   / n_samples,
            "train/top1_acc": total_top1  / n_samples,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.attn_pool.eval()

        total_loss = total_acc = total_top1 = 0.0
        n_samples = 0

        for batch in self.val_loader:
            n_groups_list = [s["n_groups"] for s in batch]

            if len(set(n_groups_list)) == 1:
                X_list = [
                    self.feature_builder(h=s["h"].to(self.device), groups=s["groups"])
                    for s in batch
                ]
                X_batch = torch.stack(X_list)
                B, M, D = X_batch.shape
                logits_batch = self.model(X_batch.view(B * M, D)).view(B, M)
            else:
                logits_batch = None

            for i, sample in enumerate(batch):
                if logits_batch is not None:
                    logits = logits_batch[i]
                else:
                    X = self.feature_builder(
                        h=sample["h"].to(self.device), groups=sample["groups"]
                    )
                    logits = self.model(X)
                pairs    = sample["pairs"].to(self.device)
                best_idx = sample["best_idx"]

                total_loss  += self._compute_loss(logits, pairs).item()
                total_acc   += compute_ranking_accuracy(logits, pairs)
                total_top1  += float(compute_top1_accuracy(logits, best_idx))
                n_samples   += 1

        return {
            "val/loss":     total_loss  / n_samples,
            "val/pair_acc": total_acc   / n_samples,
            "val/top1_acc": total_top1  / n_samples,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: str = "checkpoint"):
        path = self.save_dir / f"{name}.pt"
        torch.save({
            "epoch":                  self.epoch,
            "global_step":            self.global_step,
            "model_state_dict":       self.model.state_dict(),
            "attn_pool_state_dict":   self.attn_pool.state_dict(),
            "optimizer_state_dict":   self.optimizer.state_dict(),
            "scheduler_state_dict":   self.scheduler.state_dict(),
            "best_val_acc":           self.best_val_acc,
            "config":                 vars(self.config),
            "switcher_config":        self._sw_cfg.to_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.attn_pool.load_state_dict(ckpt["attn_pool_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch        = ckpt["epoch"]
        self.global_step  = ckpt["global_step"]
        self.best_val_acc = ckpt["best_val_acc"]
        print(f"Resumed from {path} (epoch {self.epoch})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"\nTraining for {self.config.epochs} epochs | device={self.device} | "
              f"save={self.save_dir}\n")

        for epoch in range(self.config.epochs):
            self.epoch = epoch + 1
            t0 = time.time()

            train_metrics = self.train_epoch()
            for k, v in train_metrics.items():
                self.writer.add_scalar(k, v, self.epoch)

            val_metrics = {}
            if self.epoch % self.config.eval_every == 0:
                val_metrics = self.validate()
                for k, v in val_metrics.items():
                    self.writer.add_scalar(k, v, self.epoch)

                if val_metrics["val/top1_acc"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["val/top1_acc"]
                    self.save_checkpoint("best")

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar("train/lr", lr, self.epoch)

            line = (f"Epoch {self.epoch:3d}/{self.config.epochs} | "
                    f"Loss: {train_metrics['train/loss']:.4f} | "
                    f"PairAcc: {train_metrics['train/pair_acc']:.1%} | "
                    f"Top1: {train_metrics['train/top1_acc']:.1%} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {time.time()-t0:.1f}s")
            if val_metrics:
                line += (f" | ValLoss: {val_metrics['val/loss']:.4f} | "
                         f"ValTop1: {val_metrics['val/top1_acc']:.1%}")
            print(line)

            if self.epoch % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{self.epoch}")

        self.save_checkpoint("final")
        self.writer.close()
        print(f"\nDone. Best val Top-1: {self.best_val_acc:.1%}  |  {self.save_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = CONFIG

    print("=" * 60)
    print("GroupSwitcher Training  (Attention Pooling)")
    print("=" * 60)
    for k in ("data_path", "save_dir", "epochs", "batch_size", "lr", "loss_type", "device"):
        print(f"  {k}: {cfg[k]}")
    print("=" * 60 + "\n")

    config = TrainingConfig(
        data_path=cfg["data_path"],
        embed_dim=cfg.get("embed_dim", 512),
        switcher_config_path=cfg["switcher_config_path"],
        embed_hidden=cfg["embed_hidden"],
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

    trainer = SwitcherAttnTrainer(config)

    if cfg.get("resume"):
        trainer.load_checkpoint(cfg["resume"])

    trainer.train()


if __name__ == "__main__":
    main()
