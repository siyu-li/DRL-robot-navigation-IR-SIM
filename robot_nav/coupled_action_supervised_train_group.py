"""
Supervised Training Script for Group-based Coupled Action Policy.

This script trains the v_head to predict group-specific shared velocity
using data from the obstacle-aware MARL model (marl_train_obstacle_6robots).

Key Features:
- Loads data from saved replay buffers (avoids YAML serialization issues)
- Supports predefined groups or all combinations of size 2 and 3
- Uses obstacle-aware attention encoder (iga_obstacle)
- Group-specific label: v_label = aggregate(v_group_robots)

Usage:
    Edit the CONFIG dictionary below, then run:
    python -m robot_nav.coupled_action_supervised_train_group
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from robot_nav.models.MARL.marlTD3.coupled_action_policy_obstacle import (
    CoupledActionPolicyObstacle,
)
from robot_nav.models.MARL.marlTD3.supervised_dataset_group import (
    SupervisedGroupDatasetGenerator,
    create_group_dataloader,
    generate_all_groups,
    V_LABEL_MODES,
)


# ============================================================================
# CONFIGURATION - Edit these values directly
# ============================================================================
CONFIG = {
    # Data source: path to saved replay buffer pickle file
    # The buffer is saved during marl_train_obstacle_6robots training
    "buffer_path": "robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots/replay_buffer_epoch1000.pkl",
    
    # Pretrained model from marl_train_obstacle_6robots
    "pretrained_model_name": "TD3-MARL-obstacle-6robots",
    "pretrained_directory": "robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots",
    
    # Group configuration
    # Option 1: Use all combinations of specified sizes
    "use_all_group_combinations": True,
    "group_sizes": [2, 3],  # Generate all groups of size 2 and 3
    
    # Option 2: Or specify predefined groups (set use_all_group_combinations=False)
    "predefined_groups": [
        [0, 1],      # Group 1: robots 0 and 1
        [2, 3, 4],   # Group 2: robots 2, 3, and 4
        [1, 3, 5],   # Group 3: robots 1, 3, and 5
    ],
    
    # Training hyperparameters
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-4,
    "train_split": 0.9,
    
    # Data generation options
    "sample_fraction": 0.2,  # Fraction of buffer to use (1.0 = all)
    "max_samples_per_group": 3000,  # None = no limit
    "skip_terminal": True,  # Skip terminal states (collisions)
    "uniform_sampling": True,  # If True, sample same count per group
    "samples_per_group": 5000,  # Used if uniform_sampling=True
    
    # V-label configuration
    "v_label_mode": "mean",  # "p10", "p20", "p30", "mean", "min", "max"
    "v_min": 0.0,
    "v_max": 0.5,
    
    # Freezing configuration
    "freeze_encoder": True,  # Freeze attention encoder
    "freeze_omega": True,  # Freeze omega head (angular velocity)
    "load_pretrained": True,  # Load pretrained encoder from TD3Obstacle
    
    # Model configuration
    "num_robots": 6,
    "num_obstacles": 4,  # Match your environment
    "state_dim": 11,
    "obstacle_state_dim": 4,
    "embedding_dim": 256,
    "pooling": "mean",  # "mean" or "max"
    
    # Output configuration
    "model_name": "coupled_action_group_obstacle",
    "save_directory": "robot_nav/models/MARL/marlTD3/checkpoint/group_policy",
    "save_every": 10,
    "log_every": 10,
}
# ============================================================================


def train_epoch(
    policy: CoupledActionPolicyObstacle,
    train_loader,
    epoch: int,
    log_every: int = 10
) -> float:
    """Train for one epoch."""
    policy.actor.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (robot_states, obstacle_states, group_indices, v_labels) in enumerate(pbar):
        robot_states = robot_states.to(policy.device)
        obstacle_states = obstacle_states.to(policy.device)
        group_indices = group_indices.to(policy.device)
        v_labels = v_labels.to(policy.device)
        
        loss = policy.train_step_supervised(
            robot_states, obstacle_states, group_indices, v_labels
        )
        total_loss += loss
        num_batches += 1
        
        if batch_idx % log_every == 0:
            pbar.set_postfix({"loss": f"{loss:.6f}"})
    
    return total_loss / num_batches


def validate(
    policy: CoupledActionPolicyObstacle,
    val_loader,
    epoch: int
) -> tuple:
    """Validate the model."""
    policy.actor.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    all_groups = []
    
    with torch.no_grad():
        for robot_states, obstacle_states, group_indices, v_labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            robot_states = robot_states.to(policy.device)
            obstacle_states = obstacle_states.to(policy.device)
            group_indices = group_indices.to(policy.device)
            v_labels = v_labels.to(policy.device)
            
            # Predict v_shared using batch group indices
            v_pred = policy.actor.get_v_shared_batch(
                robot_states, obstacle_states, group_indices
            )
            
            # Compute metrics
            loss = F.mse_loss(v_pred, v_labels)
            mae = torch.abs(v_pred - v_labels).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            all_preds.append(v_pred.cpu().numpy())
            all_targets.append(v_labels.cpu().numpy())
            all_groups.append(group_indices.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, avg_mae, all_preds, all_targets


def main():
    """Main training function."""
    config = CONFIG
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Determine groups to use
    if config["use_all_group_combinations"]:
        groups = generate_all_groups(config["num_robots"], config["group_sizes"])
        logger.info(f"Using all group combinations: {len(groups)} groups")
    else:
        groups = config["predefined_groups"]
        logger.info(f"Using predefined groups: {groups}")
    
    # Generate dataset from replay buffer
    logger.info("Generating supervised dataset from replay buffer...")
    generator = SupervisedGroupDatasetGenerator(
        num_robots=config["num_robots"],
        state_dim=config["state_dim"],
        obstacle_state_dim=config["obstacle_state_dim"],
        v_label_mode=config["v_label_mode"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        groups=groups,
    )
    
    buffer_path = Path(config["buffer_path"])
    if not buffer_path.exists():
        logger.error(f"Buffer file not found: {buffer_path}")
        logger.info("Please run marl_train_obstacle_6robots first to generate a replay buffer.")
        logger.info("The buffer is saved every 1000 epochs to the checkpoint directory.")
        return
    
    if config["uniform_sampling"]:
        robot_states, obstacle_states, group_indices, v_labels = generator.generate_uniform_group_samples(
            buffer_or_path=buffer_path,
            samples_per_group=config["samples_per_group"],
            skip_terminal=config["skip_terminal"],
        )
    else:
        robot_states, obstacle_states, group_indices, v_labels = generator.generate_from_buffer(
            buffer_or_path=buffer_path,
            skip_terminal=config["skip_terminal"],
            max_samples_per_group=config["max_samples_per_group"],
            sample_fraction=config["sample_fraction"],
        )
    
    if len(robot_states) == 0:
        logger.error("No samples generated. Check buffer path and contents.")
        return
    
    # Create data loaders
    train_loader, val_loader = create_group_dataloader(
        robot_states=robot_states,
        obstacle_states=obstacle_states,
        group_indices=group_indices,
        v_labels=v_labels,
        batch_size=config["batch_size"],
        shuffle=True,
        train_split=config["train_split"],
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create policy
    logger.info("Creating coupled action policy with obstacle awareness...")
    policy = CoupledActionPolicyObstacle(
        state_dim=config["state_dim"],
        obstacle_state_dim=config["obstacle_state_dim"],
        num_robots=config["num_robots"],
        num_obstacles=config["num_obstacles"],
        device=device,
        embedding_dim=config["embedding_dim"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        pooling=config["pooling"],
        load_pretrained_encoder=config["load_pretrained"],
        pretrained_model_name=config["pretrained_model_name"],
        pretrained_directory=Path(config["pretrained_directory"]),
        freeze_encoder=config["freeze_encoder"],
        freeze_omega=config["freeze_omega"],
        model_name=config["model_name"],
        save_directory=Path(config["save_directory"]),
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
            log_every=config["log_every"],
        )
        
        # Validate
        val_loss, val_mae, preds, targets = validate(
            policy=policy,
            val_loader=val_loader,
            epoch=epoch,
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
