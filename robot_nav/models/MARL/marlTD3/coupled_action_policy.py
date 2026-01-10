"""
Coupled Action Policy for Multi-Robot Navigation.

This module implements a policy where:
- All robots share ONE linear velocity v_shared.
- Each robot has its own angular velocity w_i.

The shared velocity is predicted from a global embedding G computed via
permutation-invariant pooling (mean) over per-robot embeddings H = {h_i}.
"""

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.Attention.g2anet import G2ANet
from robot_nav.models.MARL.Attention.iga import Attention


class SharedVelocityHead(nn.Module):
    """
    MLP head that predicts a shared linear velocity from a global embedding.
    
    The output is constrained to [v_min, v_max] using sigmoid scaling.
    
    Args:
        input_dim (int): Dimension of the global embedding G.
        hidden_dim (int): Hidden layer dimension.
        v_min (float): Minimum linear velocity.
        v_max (float): Maximum linear velocity.
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
            nn.Sigmoid(),  # Output in [0, 1], then scale to [v_min, v_max]
        )
    
    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict shared linear velocity from global embedding.
        
        Args:
            global_embedding (Tensor): Global embedding G of shape (B, input_dim).
            
        Returns:
            Tensor: Shared velocity v_shared of shape (B, 1) in [v_min, v_max].
        """
        raw_output = self.mlp(global_embedding)  # (B, 1) in [0, 1]
        v_shared = self.v_min + raw_output * (self.v_max - self.v_min)
        return v_shared


class CoupledActionActor(nn.Module):
    """
    Actor network for coupled action policy.
    
    Computes:
    - Per-robot embeddings H = {h_i} via GAT encoder
    - Global embedding G = mean(H) via permutation-invariant pooling
    - Shared linear velocity v_shared = v_head(G)
    - Per-robot angular velocities w_i = omega_head(h_i)
    
    Args:
        embedding_dim (int): Dimension of per-robot embeddings from GAT.
        attention (str): Attention mechanism type, one of {"igs", "g2anet"}.
        v_min (float): Minimum linear velocity for v_shared.
        v_max (float): Maximum linear velocity for v_shared.
        pooling (str): Pooling method for computing G, one of {"mean", "max"}.
        use_original_omega_head (bool): If True, uses the same architecture as the
            original policy_head (for weight loading compatibility).
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        attention: str = "igs",
        v_min: float = 0.0,
        v_max: float = 0.5,
        pooling: str = "mean",
        use_original_omega_head: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.use_original_omega_head = use_original_omega_head
        
        # GAT encoder (outputs embedding of dim = embedding_dim * 2)
        if attention == "igs":
            self.attention = Attention(embedding_dim)
        elif attention == "g2anet":
            self.attention = G2ANet(embedding_dim)
        else:
            raise ValueError(f"Unknown attention mechanism: {attention}")
        
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
            # Use the SAME architecture as original policy_head for weight loading
            # Original: Linear(512, 400) -> LeakyReLU -> Linear(400, 300) -> LeakyReLU -> Linear(300, 2) -> Tanh
            # We keep the same structure, output 2D but only use the second dimension (omega)
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
                nn.Tanh(),  # Output in [-1, 1]
            )
    
    def forward(
        self, 
        obs: torch.Tensor, 
        detach_attn: bool = False,
        return_embeddings: bool = False
    ):
        """
        Forward pass of the coupled action actor.
        
        Args:
            obs (Tensor): Observations of shape (B, N, obs_dim) or (N, obs_dim).
            detach_attn (bool): If True, detaches attention features before heads.
            return_embeddings (bool): If True, also returns per-robot and global embeddings.
            
        Returns:
            tuple containing:
                - v_shared (Tensor): Shared linear velocity, shape (B, 1).
                - omega (Tensor): Per-robot angular velocities, shape (B*N, 1).
                - action (Tensor): Combined actions [v_shared broadcast, omega], shape (B*N, 2).
                - hard_logits, pair_d, mean_entropy, hard_weights, combined_weights: Attention outputs.
                - (optional) H: Per-robot embeddings, shape (B*N, per_robot_dim).
                - (optional) G: Global embedding, shape (B, per_robot_dim).
        """
        # Handle 2D input (N, obs_dim) -> (1, N, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        
        batch_size, n_agents, _ = obs.shape
        
        # Get per-robot embeddings H from attention encoder
        H, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = (
            self.attention(obs)
        )  # H shape: (B*N, per_robot_dim)
        
        if detach_attn:
            H = H.detach()
        
        # Reshape H to (B, N, per_robot_dim) for pooling
        H_reshaped = H.view(batch_size, n_agents, self.per_robot_dim)
        
        # Compute global embedding G via permutation-invariant pooling
        if self.pooling == "mean":
            G = H_reshaped.mean(dim=1)  # (B, per_robot_dim)
        elif self.pooling == "max":
            G = H_reshaped.max(dim=1)[0]  # (B, per_robot_dim)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Predict shared linear velocity
        v_shared = self.v_head(G)  # (B, 1) in [v_min, v_max]
        
        # Predict per-robot angular velocities
        omega_output = self.omega_head(H)  # (B*N, 1) or (B*N, 2) depending on architecture
        
        # If using original architecture, extract only the angular velocity (second dim)
        if self.use_original_omega_head:
            omega = omega_output[:, 1:2]  # (B*N, 1) - second dimension is angular velocity
        else:
            omega = omega_output  # (B*N, 1)
        
        # Broadcast v_shared to all robots and combine with omega
        # v_shared: (B, 1) -> (B, N, 1) -> (B*N, 1)
        v_broadcast = v_shared.unsqueeze(1).expand(-1, n_agents, -1).reshape(-1, 1)
        action = torch.cat([v_broadcast, omega], dim=-1)  # (B*N, 2)
        
        if return_embeddings:
            return (
                v_shared, omega, action, 
                hard_logits, pair_d, mean_entropy, hard_weights, combined_weights,
                H, G
            )
        
        return (
            v_shared, omega, action,
            hard_logits, pair_d, mean_entropy, hard_weights, combined_weights
        )
    
    def get_v_shared_only(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute only the shared linear velocity (for supervised training).
        
        Args:
            obs (Tensor): Observations of shape (B, N, obs_dim) or (N, obs_dim).
            
        Returns:
            Tensor: Shared velocity v_shared of shape (B, 1).
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        
        batch_size, n_agents, _ = obs.shape
        
        # Get per-robot embeddings
        H, *_ = self.attention(obs)
        
        # Reshape and pool
        H_reshaped = H.view(batch_size, n_agents, self.per_robot_dim)
        
        if self.pooling == "mean":
            G = H_reshaped.mean(dim=1)
        elif self.pooling == "max":
            G = H_reshaped.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Predict v_shared
        v_shared = self.v_head(G)
        
        return v_shared


class CoupledActionPolicy:
    """
    Coupled Action Policy wrapper for multi-robot navigation.
    
    This policy produces:
    - A shared linear velocity for all robots
    - Per-robot angular velocities
    
    Supports loading pretrained encoder and omega head weights from decentralized policy.
    
    Args:
        state_dim (int): Per-robot state dimension.
        num_robots (int): Number of robots.
        device (torch.device): Device for computation.
        embedding_dim (int): Dimension of per-robot embeddings.
        attention (str): Attention mechanism type.
        v_min (float): Minimum linear velocity.
        v_max (float): Maximum linear velocity.
        pooling (str): Pooling method for global embedding.
        load_pretrained_encoder (bool): Whether to load pretrained encoder weights.
        pretrained_model_name (str): Name of pretrained model file.
        pretrained_directory (Path): Directory containing pretrained weights.
        freeze_encoder (bool): Whether to freeze encoder during training.
        freeze_omega (bool): Whether to freeze omega head during training.
        model_name (str): Name for this model (for logging/saving).
        save_directory (Path): Directory for saving checkpoints.
        use_original_omega_head (bool): If True, use the same omega head architecture
            as the original policy_head (for weight loading compatibility).
    """
    
    def __init__(
        self,
        state_dim: int = 11,
        num_robots: int = 5,
        device: torch.device = None,
        embedding_dim: int = 256,
        attention: str = "igs",
        v_min: float = 0.0,
        v_max: float = 0.5,
        pooling: str = "mean",
        load_pretrained_encoder: bool = False,
        pretrained_model_name: Optional[str] = None,
        pretrained_directory: Optional[Path] = None,
        freeze_encoder: bool = True,
        freeze_omega: bool = True,
        model_name: str = "coupled_action_policy",
        save_directory: Path = Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        use_original_omega_head: bool = True,
    ):
        self.state_dim = state_dim
        self.num_robots = num_robots
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.save_directory = save_directory
        self.v_min = v_min
        self.v_max = v_max
        
        # Create actor
        self.actor = CoupledActionActor(
            embedding_dim=embedding_dim,
            attention=attention,
            v_min=v_min,
            v_max=v_max,
            pooling=pooling,
            use_original_omega_head=use_original_omega_head
        ).to(self.device)
        
        # Load pretrained weights if specified
        if load_pretrained_encoder:
            if pretrained_model_name is None or pretrained_directory is None:
                raise ValueError(
                    "pretrained_model_name and pretrained_directory must be provided "
                    "when load_pretrained_encoder=True"
                )
            self.load_pretrained_weights(
                filename=pretrained_model_name,
                directory=pretrained_directory,
                load_omega=True  # Also load omega head weights
            )
        
        # Freeze parameters as specified
        if freeze_encoder:
            self._freeze_encoder()
        if freeze_omega:
            self._freeze_omega()
        
        # Setup optimizer (only for v_head by default)
        self._setup_optimizer()
        
        # Tensorboard writer
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
    
    def _freeze_encoder(self):
        """Freeze GAT encoder parameters."""
        for param in self.actor.attention.parameters():
            param.requires_grad = False
        print("Encoder (attention) parameters frozen")
    
    def _freeze_omega(self):
        """Freeze omega head parameters."""
        for param in self.actor.omega_head.parameters():
            param.requires_grad = False
        print("Omega head parameters frozen")
    
    def _unfreeze_encoder(self):
        """Unfreeze GAT encoder parameters."""
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
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    def load_pretrained_weights(
        self,
        filename: str,
        directory: Path,
        load_omega: bool = True
    ):
        """
        Load pretrained encoder (and optionally omega head) weights from decentralized policy.
        
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
            # Map policy_head weights to omega_head
            # When using original architecture, the omega_head has the SAME structure
            # as policy_head, so we can directly load the weights
            omega_keys = {
                k.replace("policy_head.", "omega_head."): v 
                for k, v in pretrained_state.items() 
                if k.startswith("policy_head.")
            }
            current_state.update(omega_keys)
            print(f"Loaded omega_head weights from policy_head ({len(omega_keys)} keys)")
        elif load_omega:
            # Due to architecture differences, we skip loading omega_head weights
            print("Note: omega_head architecture differs from policy_head; using fresh initialization")
        
        self.actor.load_state_dict(current_state, strict=False)
        print(f"Loaded pretrained encoder from: {pretrained_path}")
        print(f"  Attention keys loaded: {len(attention_keys)}")
    
    def get_action(self, obs: np.ndarray, add_noise: bool = False) -> tuple:
        """
        Compute actions for given observations.
        
        Args:
            obs (np.ndarray): Observations of shape (N, state_dim) or (B, N, state_dim).
            add_noise (bool): If True, adds exploration noise.
            
        Returns:
            tuple: (actions, v_shared, omega)
                - actions: shape (N, 2) with [v_shared, omega_i] per robot
                - v_shared: scalar shared velocity
                - omega: per-robot angular velocities
        """
        state = torch.Tensor(obs).to(self.device)
        
        with torch.no_grad():
            v_shared, omega, action, *_ = self.actor(state)
        
        action = action.cpu().numpy()
        
        if add_noise:
            # Add noise only to angular velocity (omega), not to v_shared
            noise = np.random.normal(0, 0.2, size=(action.shape[0], 1))
            action[:, 1:2] = np.clip(action[:, 1:2] + noise, -1, 1)
        
        return action.reshape(-1, 2), v_shared.cpu().numpy().item(), omega.cpu().numpy()
    
    def train_step_supervised(
        self,
        batch_states: torch.Tensor,
        batch_v_labels: torch.Tensor
    ) -> float:
        """
        Single supervised training step for v_head.
        
        Args:
            batch_states (Tensor): States of shape (B, N, state_dim).
            batch_v_labels (Tensor): Target v_shared labels of shape (B, 1).
            
        Returns:
            float: MSE loss value.
        """
        self.actor.train()
        
        # Predict v_shared
        v_pred = self.actor.get_v_shared_only(batch_states)
        
        # MSE loss
        loss = F.mse_loss(v_pred, batch_v_labels)
        
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filename: Optional[str] = None, directory: Optional[Path] = None):
        """Save model checkpoint."""
        filename = filename or self.model_name
        directory = directory or self.save_directory
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        torch.save(
            self.actor.state_dict(),
            f"{directory}/{filename}_actor.pth"
        )
        print(f"Saved model to: {directory}/{filename}_actor.pth")
    
    def load(self, filename: str, directory: Path):
        """Load model checkpoint."""
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        print(f"Loaded model from: {directory}/{filename}_actor.pth")
    
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
        Convert raw environment outputs into per-agent state vectors.
        
        Matches the format used in decentralized/centralized policies.
        
        Args:
            poses (list): Per-agent poses [[x, y, theta], ...].
            distance (list): Per-agent distances to goal.
            cos (list): Per-agent cos(heading error to goal).
            sin (list): Per-agent sin(heading error to goal).
            collision (list): Per-agent collision flags.
            action (list): Per-agent last actions [[lin_vel, ang_vel], ...].
            goal_positions (list): Per-agent goals [[gx, gy], ...].
            
        Returns:
            list: Per-agent state vectors.
        """
        states = []
        
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
        
        return states
