"""
Coupled Action Policy with Obstacle Awareness for Group-based Multi-Robot Navigation.

This module extends coupled_action_policy.py to:
1. Support obstacle graph nodes (like marlTD3_obstacle.py)
2. Use group-based embedding pooling for v_shared
3. Work with the obstacle-aware attention mechanism (iga_obstacle.py)

Key Architecture:
- Uses AttentionObstacle instead of Attention for obstacle-aware embeddings
- Pools embeddings only from robots in the active group for G
- Outputs v_shared for the group and per-robot omega
"""

from pathlib import Path
from typing import Optional, Literal, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.Attention.iga_obstacle import AttentionObstacle


class SharedVelocityHead(nn.Module):
    """
    MLP head that predicts a shared linear velocity from a global embedding.
    
    The output is constrained to [v_min, v_max] using sigmoid scaling.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128,
        v_min: float = 0.0,
        v_max: float = 0.5
    ):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict shared linear velocity from global embedding.
        
        Args:
            global_embedding (Tensor): Global embedding G of shape (B, input_dim).
            
        Returns:
            Tensor: Shared velocity v_shared of shape (B, 1) in [v_min, v_max].
        """
        raw_output = self.mlp(global_embedding)
        v_shared = self.v_min + raw_output * (self.v_max - self.v_min)
        return v_shared


class CoupledActionActorObstacle(nn.Module):
    """
    Actor network for coupled action policy with obstacle awareness.
    
    Computes:
    - Per-robot embeddings H via obstacle-aware GAT encoder
    - Group embedding G = mean(H_group) via pooling over group robots only
    - Shared linear velocity v_shared = v_head(G)
    - Per-robot angular velocities w_i = omega_head(h_i)
    
    Args:
        embedding_dim (int): Dimension of per-robot embeddings from GAT.
        v_min (float): Minimum linear velocity for v_shared.
        v_max (float): Maximum linear velocity for v_shared.
        pooling (str): Pooling method for computing G, one of {"mean", "max"}.
        use_original_omega_head (bool): If True, uses the same architecture as the
            original policy_head (for weight loading compatibility).
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        v_min: float = 0.0,
        v_max: float = 0.5,
        pooling: str = "mean",
        use_original_omega_head: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.use_original_omega_head = use_original_omega_head
        
        # Obstacle-aware GAT encoder
        self.attention = AttentionObstacle(embedding_dim)
        
        # Per-robot embedding dimension after attention
        self.per_robot_dim = embedding_dim * 2  # 512 for embedding_dim=256
        
        # Shared velocity head: G -> v_shared
        self.v_head = SharedVelocityHead(
            input_dim=self.per_robot_dim,
            hidden_dim=128,
            v_min=v_min,
            v_max=v_max
        )
        
        # Per-robot angular velocity head: h_i -> w_i
        if use_original_omega_head:
            # Match original policy_head architecture for weight loading
            self.omega_head = nn.Sequential(
                nn.Linear(self.per_robot_dim, 400),
                nn.LeakyReLU(),
                nn.Linear(400, 300),
                nn.LeakyReLU(),
                nn.Linear(300, 2),  # Output both v and omega for compatibility
                nn.Tanh(),
            )
        else:
            # Lighter architecture for fresh training
            self.omega_head = nn.Sequential(
                nn.Linear(self.per_robot_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 1),
                nn.Tanh(),
            )
    
    def _mask_inactive_robot_features(
        self, 
        robot_obs: torch.Tensor, 
        active_group: List[int]
    ) -> torch.Tensor:
        """
        Mask non-active robots' features to treat them like obstacles.
        
        Zeros out velocity (indices 7:9), goal (indices 9:11), and goal-related
        features (indices 4:7) for robots not in the active group.
        
        State layout: [px, py, cos_h, sin_h, dist/17, cos_err, sin_err, lin_v, ang_v, gx, gy]
                       0   1    2      3       4        5        6        7      8    9   10
        
        Args:
            robot_obs (Tensor): Robot observations of shape (B, N_robots, state_dim).
            active_group (List[int]): Indices of robots in the active group.
            
        Returns:
            Tensor: Masked robot observations with same shape as input.
        """
        masked_obs = robot_obs.clone()
        n_robots = masked_obs.shape[1]
        
        for i in range(n_robots):
            if i not in active_group:
                # Zero out velocity (lin_v, ang_v) at indices 7:9
                masked_obs[:, i, 7:9] = 0.0
                # Zero out goal (gx, gy) at indices 9:11
                masked_obs[:, i, 9:11] = 0.0
                # Zero out goal-related features: dist/17, cos_err, sin_err at indices 4:7
                masked_obs[:, i, 4:7] = 0.0
        
        return masked_obs
    
    def _mask_inactive_robot_features_batch(
        self, 
        robot_obs: torch.Tensor, 
        group_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask non-active robots' features for batched inputs with per-sample groups.
        
        Args:
            robot_obs (Tensor): Robot observations of shape (B, N_robots, state_dim).
            group_indices (Tensor): Per-sample group indices of shape (B, max_group_size).
                Padded with -1 for variable group sizes.
            
        Returns:
            Tensor: Masked robot observations with same shape as input.
        """
        batch_size, n_robots, state_dim = robot_obs.shape
        masked_obs = robot_obs.clone()
        
        for b in range(batch_size):
            # Get valid group indices (exclude -1 padding)
            group = group_indices[b]
            valid_mask = group >= 0
            valid_indices = set(group[valid_mask].tolist())
            
            for i in range(n_robots):
                if i not in valid_indices:
                    # Zero out velocity, goal, and goal-related features
                    masked_obs[b, i, 7:9] = 0.0
                    masked_obs[b, i, 9:11] = 0.0
                    masked_obs[b, i, 4:7] = 0.0
        
        return masked_obs
    
    def forward(
        self, 
        robot_obs: torch.Tensor,
        obstacle_obs: torch.Tensor,
        detach_attn: bool = False,
        return_embeddings: bool = False,
        active_group: Optional[List[int]] = None,
    ):
        """
        Forward pass of the coupled action actor with obstacle awareness.
        
        Args:
            robot_obs (Tensor): Robot observations of shape (B, N_robots, robot_state_dim) or (N_robots, robot_state_dim).
            obstacle_obs (Tensor): Obstacle observations of shape (B, N_obs, obstacle_state_dim) or (N_obs, obstacle_state_dim).
            detach_attn (bool): If True, detaches attention features before heads.
            return_embeddings (bool): If True, also returns per-robot and global embeddings.
            active_group (List[int], optional): List of robot indices in active group.
                If None, all robots are considered active. When specified, non-active
                robots' velocity and goal features are zeroed out (like obstacles).
            
        Returns:
            tuple containing:
                - v_shared (Tensor): Shared linear velocity, shape (B, 1).
                - omega (Tensor): Per-robot angular velocities, shape (B*N_robots, 1).
                - action (Tensor): Combined actions [v_shared broadcast, omega], shape (B*N_robots, 2).
                - Attention outputs for logging.
                - (optional) H: Per-robot embeddings, shape (B, N_robots, per_robot_dim).
                - (optional) G: Global embedding, shape (B, per_robot_dim).
        """
        # Handle 2D input
        if robot_obs.dim() == 2:
            robot_obs = robot_obs.unsqueeze(0)
            obstacle_obs = obstacle_obs.unsqueeze(0)
        
        batch_size, n_robots, _ = robot_obs.shape
        
        # Mask non-active robots' features when a group is specified
        if active_group is not None and len(active_group) > 0:
            robot_obs = self._mask_inactive_robot_features(robot_obs, active_group)
        
        # Get per-robot embeddings H from obstacle-aware attention encoder
        (
            H,
            hard_logits_rr, hard_logits_ro,
            dist_rr, dist_ro,
            mean_entropy,
            hard_weights_rr, hard_weights_ro,
            combined_weights,
        ) = self.attention(robot_obs, obstacle_obs)
        # H shape: (B*N_robots, per_robot_dim)
        
        if detach_attn:
            H = H.detach()
        
        # Reshape H to (B, N_robots, per_robot_dim) for pooling
        H_reshaped = H.view(batch_size, n_robots, self.per_robot_dim)
        
        # Compute group embedding G via pooling over active group only
        if active_group is not None and len(active_group) > 0:
            active_indices = torch.tensor(active_group, device=H.device, dtype=torch.long)
            H_active = H_reshaped[:, active_indices, :]  # (B, group_size, per_robot_dim)
            
            if self.pooling == "mean":
                G = H_active.mean(dim=1)  # (B, per_robot_dim)
            elif self.pooling == "max":
                G = H_active.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
        else:
            # Pool over all robots
            if self.pooling == "mean":
                G = H_reshaped.mean(dim=1)
            elif self.pooling == "max":
                G = H_reshaped.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Predict shared linear velocity
        v_shared = self.v_head(G)  # (B, 1) in [v_min, v_max]
        
        # Predict per-robot angular velocities
        omega_output = self.omega_head(H)  # (B*N_robots, 1) or (B*N_robots, 2)
        
        if self.use_original_omega_head:
            omega = omega_output[:, 1:2]  # Extract angular velocity
        else:
            omega = omega_output
        
        # Broadcast v_shared to all robots
        v_broadcast = v_shared.unsqueeze(1).expand(-1, n_robots, -1).reshape(-1, 1)
        action = torch.cat([v_broadcast, omega], dim=-1)  # (B*N_robots, 2)
        
        # Apply active group mask: zero out actions for inactive robots
        if active_group is not None:
            active_mask = torch.zeros(n_robots, device=H.device)
            active_mask[active_group] = 1.0
            active_mask = active_mask.unsqueeze(0).expand(batch_size, -1).reshape(-1, 1)
            action = action * active_mask
        
        attention_outputs = (
            hard_logits_rr, hard_logits_ro,
            dist_rr, dist_ro,
            mean_entropy,
            hard_weights_rr, hard_weights_ro,
            combined_weights,
        )
        
        if return_embeddings:
            return v_shared, omega, action, attention_outputs, H_reshaped, G
        
        return v_shared, omega, action, attention_outputs
    
    def get_v_shared_only(
        self, 
        robot_obs: torch.Tensor,
        obstacle_obs: torch.Tensor,
        active_group: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute only the shared linear velocity (for supervised training).
        
        Args:
            robot_obs (Tensor): Robot observations.
            obstacle_obs (Tensor): Obstacle observations.
            active_group (List[int], optional): Indices of robots in the group.
                When specified, non-active robots' features are zeroed out.
            
        Returns:
            Tensor: Shared velocity v_shared of shape (B, 1).
        """
        if robot_obs.dim() == 2:
            robot_obs = robot_obs.unsqueeze(0)
            obstacle_obs = obstacle_obs.unsqueeze(0)
        
        batch_size, n_robots, _ = robot_obs.shape
        
        # Mask non-active robots' features when a group is specified
        if active_group is not None and len(active_group) > 0:
            robot_obs = self._mask_inactive_robot_features(robot_obs, active_group)
        
        # Get per-robot embeddings
        H, *_ = self.attention(robot_obs, obstacle_obs)
        
        # Reshape and pool
        H_reshaped = H.view(batch_size, n_robots, self.per_robot_dim)
        
        if active_group is not None and len(active_group) > 0:
            active_indices = torch.tensor(active_group, device=H.device, dtype=torch.long)
            H_active = H_reshaped[:, active_indices, :]
            
            if self.pooling == "mean":
                G = H_active.mean(dim=1)
            elif self.pooling == "max":
                G = H_active.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
        else:
            if self.pooling == "mean":
                G = H_reshaped.mean(dim=1)
            elif self.pooling == "max":
                G = H_reshaped.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        v_shared = self.v_head(G)
        return v_shared
    
    def get_v_shared_batch(
        self, 
        robot_obs: torch.Tensor,
        obstacle_obs: torch.Tensor,
        group_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute v_shared for a batch with per-sample group indices.
        
        This is the main method for supervised training where each sample
        in the batch may have a different group. Non-active robots' features
        are masked (zeroed out) for each sample based on its group_indices.
        
        Args:
            robot_obs (Tensor): Robot observations of shape (B, N_robots, state_dim).
            obstacle_obs (Tensor): Obstacle observations of shape (B, N_obs, obs_dim).
            group_indices (Tensor): Per-sample group indices of shape (B, max_group_size).
                Padded with -1 for variable group sizes.
            
        Returns:
            Tensor: Shared velocity v_shared of shape (B, 1).
        """
        batch_size, n_robots, _ = robot_obs.shape
        
        # Mask non-active robots' features for each sample based on its group
        robot_obs = self._mask_inactive_robot_features_batch(robot_obs, group_indices)
        
        # Get per-robot embeddings for all samples
        H, *_ = self.attention(robot_obs, obstacle_obs)
        H_reshaped = H.view(batch_size, n_robots, self.per_robot_dim)
        
        # Compute G for each sample using its specific group
        G_list = []
        for b in range(batch_size):
            # Get valid group indices (exclude -1 padding)
            group = group_indices[b]
            valid_mask = group >= 0
            valid_indices = group[valid_mask]
            
            if len(valid_indices) == 0:
                # Fallback: use all robots if group is empty
                H_group = H_reshaped[b]
            else:
                H_group = H_reshaped[b, valid_indices, :]  # (group_size, per_robot_dim)
            
            if self.pooling == "mean":
                G_b = H_group.mean(dim=0)  # (per_robot_dim,)
            elif self.pooling == "max":
                G_b = H_group.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
            
            G_list.append(G_b)
        
        G = torch.stack(G_list, dim=0)  # (B, per_robot_dim)
        v_shared = self.v_head(G)  # (B, 1)
        
        return v_shared


class CoupledActionPolicyObstacle:
    """
    Coupled Action Policy wrapper with obstacle awareness for group-based navigation.
    
    This policy produces:
    - A shared linear velocity for robots in the active group
    - Per-robot angular velocities
    
    Supports:
    - Loading pretrained encoder from TD3Obstacle model
    - Group-based embedding pooling
    - Obstacle-aware attention mechanism
    
    Args:
        state_dim (int): Per-robot state dimension.
        obstacle_state_dim (int): Per-obstacle state dimension.
        num_robots (int): Number of robots.
        num_obstacles (int): Number of obstacles.
        device (torch.device): Device for computation.
        embedding_dim (int): Dimension of per-robot embeddings.
        v_min (float): Minimum linear velocity.
        v_max (float): Maximum linear velocity.
        pooling (str): Pooling method for group embedding.
        load_pretrained_encoder (bool): Whether to load pretrained encoder weights.
        pretrained_model_name (str): Name of pretrained model file.
        pretrained_directory (Path): Directory containing pretrained weights.
        freeze_encoder (bool): Whether to freeze encoder during training.
        freeze_omega (bool): Whether to freeze omega head during training.
        model_name (str): Name for this model.
        save_directory (Path): Directory for saving checkpoints.
        use_original_omega_head (bool): If True, use original omega head architecture.
    """
    
    def __init__(
        self,
        state_dim: int = 11,
        obstacle_state_dim: int = 4,
        num_robots: int = 6,
        num_obstacles: int = 4,
        device: torch.device = None,
        embedding_dim: int = 256,
        v_min: float = 0.0,
        v_max: float = 0.5,
        pooling: str = "mean",
        load_pretrained_encoder: bool = False,
        pretrained_model_name: Optional[str] = None,
        pretrained_directory: Optional[Path] = None,
        freeze_encoder: bool = True,
        freeze_omega: bool = True,
        model_name: str = "coupled_action_obstacle",
        save_directory: Path = Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        use_original_omega_head: bool = True,
    ):
        self.state_dim = state_dim
        self.obstacle_state_dim = obstacle_state_dim
        self.num_robots = num_robots
        self.num_obstacles = num_obstacles
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.save_directory = Path(save_directory)
        self.v_min = v_min
        self.v_max = v_max
        
        # Create actor
        self.actor = CoupledActionActorObstacle(
            embedding_dim=embedding_dim,
            v_min=v_min,
            v_max=v_max,
            pooling=pooling,
            use_original_omega_head=use_original_omega_head
        ).to(self.device)
        
        # Load pretrained weights if specified
        if load_pretrained_encoder and pretrained_model_name and pretrained_directory:
            self.load_pretrained_weights(
                filename=pretrained_model_name,
                directory=pretrained_directory,
                load_omega=use_original_omega_head
            )
        
        # Freeze parameters as specified
        if freeze_encoder:
            self._freeze_encoder()
        if freeze_omega:
            self._freeze_omega()
        
        # Setup optimizer (only for trainable params)
        self._setup_optimizer()
        
        # Tensorboard writer
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
    
    def _freeze_encoder(self):
        """Freeze attention encoder parameters."""
        for param in self.actor.attention.parameters():
            param.requires_grad = False
        print("Encoder (attention) parameters frozen")
    
    def _freeze_omega(self):
        """Freeze omega head parameters."""
        for param in self.actor.omega_head.parameters():
            param.requires_grad = False
        print("Omega head parameters frozen")
    
    def _unfreeze_encoder(self):
        """Unfreeze attention encoder parameters."""
        for param in self.actor.attention.parameters():
            param.requires_grad = True
        print("Encoder (attention) parameters unfrozen")
    
    def _unfreeze_omega(self):
        """Unfreeze omega head parameters."""
        for param in self.actor.omega_head.parameters():
            param.requires_grad = True
        print("Omega head parameters unfrozen")
    
    def _setup_optimizer(self, lr: float = 1e-4):
        """Setup optimizer for trainable parameters only."""
        trainable_params = [p for p in self.actor.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        else:
            self.optimizer = None
            print("Warning: No trainable parameters!")
    
    def load_pretrained_weights(
        self,
        filename: str,
        directory: Path,
        load_omega: bool = True
    ):
        """
        Load pretrained encoder (and optionally omega head) from TD3Obstacle model.
        
        Args:
            filename (str): Base filename of pretrained model.
            directory (Path): Directory containing pretrained weights.
            load_omega (bool): Whether to also load omega head weights from policy_head.
        """
        pretrained_path = f"{directory}/{filename}_actor.pth"
        pretrained_state = torch.load(pretrained_path, map_location=self.device)
        
        # Load attention (encoder) weights
        attention_keys = {
            k: v for k, v in pretrained_state.items() 
            if k.startswith("attention.")
        }
        
        current_state = self.actor.state_dict()
        current_state.update(attention_keys)
        
        if load_omega and self.actor.use_original_omega_head:
            # Load omega head from policy_head
            omega_keys = {}
            for k, v in pretrained_state.items():
                if k.startswith("policy_head."):
                    new_key = k.replace("policy_head.", "omega_head.")
                    omega_keys[new_key] = v
            current_state.update(omega_keys)
            print(f"  Omega head keys loaded: {len(omega_keys)}")
        
        self.actor.load_state_dict(current_state, strict=False)
        print(f"Loaded pretrained encoder from: {pretrained_path}")
        print(f"  Attention keys loaded: {len(attention_keys)}")
    
    def get_action(
        self, 
        robot_obs: np.ndarray,
        obstacle_obs: np.ndarray,
        active_group: Optional[List[int]] = None,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        """
        Get action for the current observation.
        
        Args:
            robot_obs (np.ndarray): Robot observations of shape (N_robots, state_dim).
            obstacle_obs (np.ndarray): Obstacle observations of shape (N_obs, obs_dim).
            active_group (List[int], optional): Robot indices in the active group.
                When specified, non-active robots' velocity and goal features are
                zeroed out (similar to how obstacles are treated).
            add_noise (bool): Whether to add exploration noise.
            
        Returns:
            Tuple of (action, v_shared, combined_weights):
                - action: Array of shape (N_robots, 2)
                - v_shared: Scalar shared linear velocity
                - combined_weights: Attention weights for visualization
        """
        self.actor.eval()
        
        robot_state = torch.FloatTensor(robot_obs).to(self.device)
        obstacle_state = torch.FloatTensor(obstacle_obs).to(self.device)
        
        with torch.no_grad():
            v_shared, omega, action, attention_outputs = self.actor(
                robot_state, obstacle_state, active_group=active_group
            )
        
        action_np = action.cpu().numpy().reshape(-1, 2)
        v_shared_np = v_shared.cpu().numpy().item()
        combined_weights = attention_outputs[-1]  # Last element is combined_weights
        
        if add_noise:
            noise = np.random.normal(0, 0.1, size=action_np.shape)
            action_np = np.clip(action_np + noise, -1, 1)
        
        return action_np, v_shared_np, combined_weights
    
    def train_step_supervised(
        self,
        batch_robot_states: torch.Tensor,
        batch_obstacle_states: torch.Tensor,
        batch_group_indices: torch.Tensor,
        batch_v_labels: torch.Tensor,
    ) -> float:
        """
        Perform one supervised training step on v_head.
        
        Args:
            batch_robot_states (Tensor): Robot states of shape (B, N_robots, state_dim).
            batch_obstacle_states (Tensor): Obstacle states of shape (B, N_obs, obs_dim).
            batch_group_indices (Tensor): Group indices of shape (B, max_group_size).
            batch_v_labels (Tensor): Target v_shared labels of shape (B, 1).
            
        Returns:
            float: MSE loss value.
        """
        self.actor.train()
        
        # Predict v_shared using per-sample group indices
        v_pred = self.actor.get_v_shared_batch(
            batch_robot_states, 
            batch_obstacle_states,
            batch_group_indices
        )
        
        # Compute MSE loss
        loss = F.mse_loss(v_pred, batch_v_labels)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Logging
        self.writer.add_scalar("train/v_loss", loss.item(), self.iter_count)
        self.iter_count += 1
        
        return loss.item()
    
    def save(self, filename: Optional[str] = None, directory: Optional[Path] = None):
        """Save model weights."""
        if filename is None:
            filename = self.model_name
        if directory is None:
            directory = self.save_directory
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        print(f"Model saved to: {directory}/{filename}_actor.pth")
    
    def load(self, filename: str, directory: Path):
        """Load model weights."""
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        print(f"Model loaded from: {directory}/{filename}_actor.pth")
    
    def prepare_state(
        self, 
        poses, 
        distance, 
        cos, 
        sin, 
        collision, 
        action, 
        goal_positions
    ):
        """
        Convert raw environment outputs into per-robot state vectors.
        
        Use this for live inference when getting raw outputs from simulator.
        NOT needed when loading from replay buffer (states already prepared).
        
        Args:
            poses (list): Per-robot poses [[x, y, theta], ...].
            distance (list): Per-robot distances to goal.
            cos (list): Per-robot cos(heading error to goal).
            sin (list): Per-robot sin(heading error to goal).
            collision (list): Per-robot collision flags.
            action (list): Per-robot last actions [[lin_vel, ang_vel], ...].
            goal_positions (list): Per-robot goals [[gx, gy], ...].
            
        Returns:
            tuple: (states, terminal)
                - states: Per-robot state vectors, list of shape (N_robots, state_dim)
                - terminal: Per-robot terminal flags
        """
        states = []
        terminal = []
        
        for i in range(self.num_robots):
            pose = poses[i]
            goal_pos = goal_positions[i]
            act = action[i]
            
            px, py, theta = pose
            gx, gy = goal_pos
            
            heading_cos = np.cos(theta)
            heading_sin = np.sin(theta)
            lin_vel = act[0] * 2
            ang_vel = (act[1] + 1) / 2
            
            state = [
                px, py,
                heading_cos, heading_sin,
                distance[i] / 17,
                cos[i], sin[i],
                lin_vel, ang_vel,
                gx, gy,
            ]
            
            assert len(state) == self.state_dim, \
                f"State length mismatch: expected {self.state_dim}, got {len(state)}"
            states.append(state)
            terminal.append(collision[i])
        
        return states, terminal
