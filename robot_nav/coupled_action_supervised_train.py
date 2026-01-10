"""
Supervised Training Script for Coupled Action Policy v_head.

This script trains only the shared velocity head (v_head) using supervised learning
with labels generated from decentralized policy rollout data.

Usage:
    Edit the CONFIG dictionary below, then run:
    python -m robot_nav.coupled_action_supervised_train
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.coupled_action_policy import (
    CoupledActionPolicy,
    CoupledActionActor
)
from robot_nav.models.MARL.marlTD3.supervised_dataset import (
    SupervisedDatasetGenerator,
    create_dataloader,
    V_LABEL_MODES
)


# ============================================================================
# CONFIGURATION - Edit these values directly
# ============================================================================
CONFIG = {
    # Data paths
    "data_path": "robot_nav/assets/marl_data.yml",  # Can be comma-separated for multiple files
    "pretrained_model_name": "TDR-MARL-train",  # Name of pretrained decentralized model
    "pretrained_directory": "robot_nav/models/MARL/marlTD3/checkpoint",
    
    # Training hyperparameters
    "epochs": 100,  # Number of training epochs
    "batch_size": 64,  # Batch size
    "lr": 1e-4,  # Learning rate
    "train_split": 0.9,  # Train/val split ratio (0.9 = 90% train, 10% val)
    "max_samples": None,  # Max samples to use (None = use all available)
    
    # V-label configuration
    "v_label_mode": "p20",  # Aggregation mode: "p10", "p20", "p30", "mean", "min"
    "v_min": 0.0,  # Minimum linear velocity
    "v_max": 0.5,  # Maximum linear velocity
    
    # Freezing configuration
    "freeze_encoder": True,  # Freeze GAT encoder parameters during training
    "freeze_omega": True,  # Freeze omega head parameters during training
    "load_pretrained": True,  # Load pretrained encoder weights
    
    # Model configuration
    "num_robots": 5,  # Number of robots
    "state_dim": 11,  # Per-robot state dimension
    "embedding_dim": 256,  # Embedding dimension
    "attention": "igs",  # Attention mechanism: "igs" or "g2anet"
    "pooling": "mean",  # Pooling method: "mean" or "max"
    
    # Output configuration
    "model_name": "coupled_action_supervised",  # Name for saved model and logs
    "save_directory": "robot_nav/models/MARL/marlTD3/checkpoint",
    "save_every": 10,  # Save checkpoint every N epochs
    "log_every": 10,  # Log metrics every N batches
}
# ============================================================================


def train_epoch(
    policy: CoupledActionPolicy,
    train_loader,
    epoch: int,
    log_every: int = 10
) -> float:
    """
    Train for one epoch.
    
    Args:
        policy: CoupledActionPolicy instance.
        train_loader: Training DataLoader.
        epoch: Current epoch number.
        log_every: Log frequency in batches.
        
    Returns:
        Average training loss for the epoch.
    """
    policy.actor.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (states, v_labels) in enumerate(pbar):
        states = states.to(policy.device)
        v_labels = v_labels.to(policy.device)
        
        loss = policy.train_step_supervised(states, v_labels)
        total_loss += loss
        num_batches += 1
        
        if batch_idx % log_every == 0:
            pbar.set_postfix({"loss": f"{loss:.6f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    policy: CoupledActionPolicy,
    val_loader,
    epoch: int
) -> tuple:
    """
    Validate the model.
    
    Args:
        policy: CoupledActionPolicy instance.
        val_loader: Validation DataLoader.
        epoch: Current epoch number.
        
    Returns:
        Tuple of (avg_loss, avg_mae, predictions, targets).
    """
    policy.actor.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for states, v_labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            states = states.to(policy.device)
            v_labels = v_labels.to(policy.device)
            
            # Predict v_shared
            v_pred = policy.actor.get_v_shared_only(states)
            
            # Compute metrics
            loss = F.mse_loss(v_pred, v_labels)
            mae = torch.abs(v_pred - v_labels).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            all_preds.append(v_pred.cpu().numpy())
            all_targets.append(v_labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, avg_mae, all_preds, all_targets


def main():
    """Main training function."""
    # Load configuration
    config = CONFIG
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Parse data paths
    data_paths = [p.strip() for p in config["data_path"].split(",")]
    logger.info(f"Data paths: {data_paths}")
    
    # Generate supervised dataset
    logger.info("Generating supervised dataset...")
    generator = SupervisedDatasetGenerator(
        file_paths=data_paths,
        num_robots=config["num_robots"],
        state_dim=config["state_dim"],
        v_label_mode=config["v_label_mode"],
        v_min=config["v_min"],
        v_max=config["v_max"]
    )
    
    states, v_labels = generator.generate_from_yaml(
        skip_collisions=True,
        max_samples=config["max_samples"]
    )
    
    if len(states) == 0:
        logger.error("No samples generated. Check data path and format.")
        return
    
    # Create data loaders
    train_loader, val_loader = create_dataloader(
        states=states,
        v_labels=v_labels,
        batch_size=config["batch_size"],
        shuffle=True,
        train_split=config["train_split"]
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create policy
    logger.info("Creating coupled action policy...")
    policy = CoupledActionPolicy(
        state_dim=config["state_dim"],
        num_robots=config["num_robots"],
        device=device,
        embedding_dim=config["embedding_dim"],
        attention=config["attention"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        pooling=config["pooling"],
        load_pretrained_encoder=config["load_pretrained"],
        pretrained_model_name=config["pretrained_model_name"],
        pretrained_directory=Path(config["pretrained_directory"]),
        freeze_encoder=config["freeze_encoder"],
        freeze_omega=config["freeze_omega"],
        model_name=config["model_name"],
        save_directory=Path(config["save_directory"])
    )
    
    # Update optimizer with specified learning rate
    policy._setup_optimizer(lr=config["lr"])
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in policy.actor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.actor.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # TensorBoard writer
    writer = SummaryWriter(comment=f"-{config['model_name']}")
    
    # Training loop
    best_val_loss = float("inf")
    
    logger.info("Starting training...")
    for epoch in range(1, config["epochs"] + 1):
        # Train
        train_loss = train_epoch(
            policy=policy,
            train_loader=train_loader,
            epoch=epoch,
            log_every=config["log_every"]
        )
        
        # Validate
        val_loss, val_mae, preds, targets = validate(
            policy=policy,
            val_loader=val_loader,
            epoch=epoch
        )
        
        # Log metrics
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mae", val_mae, epoch)
        
        # Log prediction distribution
        writer.add_histogram("val/predictions", preds, epoch)
        writer.add_histogram("val/targets", targets, epoch)
        
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, val_mae={val_mae:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            policy.save(filename=f"{config['model_name']}_best")
            logger.info(f"  New best model saved (val_loss={val_loss:.6f})")
        
        # Periodic checkpoint
        if epoch % config["save_every"] == 0:
            policy.save(filename=f"{config['model_name']}_epoch{epoch}")
    
    # Save final model
    policy.save(filename=f"{config['model_name']}_final")
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
    writer.close()


if __name__ == "__main__":
    main()
