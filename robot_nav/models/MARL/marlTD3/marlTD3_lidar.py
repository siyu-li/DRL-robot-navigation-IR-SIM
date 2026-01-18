"""
TD3 with LiDAR Support - Late Fusion Architecture (Augmented State).

Key Design Decisions:
1. The attention network remains UNCHANGED (original IGA/G2ANet)
2. LiDAR is concatenated with raw state to form augmented state
3. State is split inside the model: agent_state (11-dim) | lidar (num_beams)
4. Attention processes agent_state only
5. LiDAR is encoded and fused AFTER attention (late fusion)

State Format:
    Augmented state = [agent_state (11-dim), lidar_scan (num_beams)]
    Total dim = 11 + num_beams (e.g., 11 + 180 = 191)

This allows reusing the original ReplayBuffer (no separate LiDAR storage).
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

# Use ORIGINAL attention modules (unchanged)
from robot_nav.models.MARL.Attention.g2anet import G2ANet
from robot_nav.models.MARL.Attention.iga import Attention
from robot_nav.models.MARL.lidar_encoder import create_lidar_encoder

# Raw agent state dimension (without LiDAR)
RAW_STATE_DIM = 11


class ActorWithLiDAR(nn.Module):
    """
    Actor network with late LiDAR fusion.

    Receives augmented state (agent_state + lidar), splits it internally,
    processes agent_state through attention, encodes lidar separately,
    then fuses for the policy head.
    """

    def __init__(
        self,
        action_dim: int,
        embedding_dim: int = 256,
        attention: str = "igs",
        use_lidar: bool = True,
        lidar_encoder_type: str = "sector",
        lidar_num_beams: int = 180,
        lidar_embed_dim: int = 12,
        lidar_encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.use_lidar = use_lidar
        self.lidar_num_beams = lidar_num_beams
        self.lidar_embed_dim = lidar_embed_dim if use_lidar else 0

        # Original attention (unchanged)
        if attention == "igs":
            self.attention = Attention(embedding_dim)
        elif attention == "g2anet":
            self.attention = G2ANet(embedding_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention}")

        # LiDAR encoder (separate from attention)
        if use_lidar:
            lidar_kwargs = lidar_encoder_kwargs or {}
            self.lidar_encoder = create_lidar_encoder(
                encoder_type=lidar_encoder_type,
                num_beams=lidar_num_beams,
                output_dim=lidar_embed_dim,
                **lidar_kwargs,
            )
        else:
            self.lidar_encoder = None

        # Policy head: attention output (512) + LiDAR embedding
        policy_input_dim = embedding_dim * 2 + self.lidar_embed_dim

        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

    def _split_state(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Split augmented state into agent_state and lidar.

        Args:
            obs: Augmented state (B, N, 11+num_beams) or (N, 11+num_beams)

        Returns:
            agent_state: (B, N, 11) or (N, 11)
            lidar: (B, N, num_beams) or (N, num_beams), or None if no LiDAR
        """
        if self.use_lidar:
            agent_state = obs[..., :RAW_STATE_DIM]
            lidar = obs[..., RAW_STATE_DIM:]
            return agent_state, lidar
        else:
            return obs, None

    def forward(self, obs: torch.Tensor, detach_attn: bool = False) -> Tuple:
        """
        Forward pass with late LiDAR fusion.

        Args:
            obs (Tensor): Augmented state (B, N, state_dim) or (N, state_dim).
                          state_dim = 11 + num_beams if use_lidar, else 11.
            detach_attn (bool): Whether to detach attention features.

        Returns:
            tuple: (action, hard_logits, pair_d, entropy, hard_weights, combined_weights)
        """
        # Split augmented state
        agent_state, lidar = self._split_state(obs)

        # Original attention on agent state only
        attn_out, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = (
            self.attention(agent_state)
        )

        if detach_attn:
            attn_out = attn_out.detach()

        # Late fusion: concatenate attention output with LiDAR embedding
        if self.use_lidar and lidar is not None:
            # Flatten lidar for encoder: (B, N, num_beams) -> (B*N, num_beams)
            if lidar.dim() == 3:
                batch_size, n_agents, num_beams = lidar.shape
                lidar_flat = lidar.reshape(-1, num_beams)
            else:
                lidar_flat = lidar

            lidar_embed = self.lidar_encoder(lidar_flat)  # (B*N, lidar_embed_dim)
            fused = torch.cat([attn_out, lidar_embed], dim=-1)
        else:
            fused = attn_out

        action = self.policy_head(fused)
        return action, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights


class CriticWithLiDAR(nn.Module):
    """
    Twin Q-value critic with late LiDAR fusion.
    """

    def __init__(
        self,
        action_dim: int,
        embedding_dim: int = 256,
        attention: str = "igs",
        use_lidar: bool = True,
        lidar_encoder_type: str = "sector",
        lidar_num_beams: int = 180,
        lidar_embed_dim: int = 12,
        lidar_encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_lidar = use_lidar
        self.lidar_num_beams = lidar_num_beams
        self.lidar_embed_dim = lidar_embed_dim if use_lidar else 0

        # Original attention (unchanged)
        if attention == "igs":
            self.attention = Attention(embedding_dim)
        elif attention == "g2anet":
            self.attention = G2ANet(embedding_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention}")

        # LiDAR encoder
        if use_lidar:
            lidar_kwargs = lidar_encoder_kwargs or {}
            self.lidar_encoder = create_lidar_encoder(
                encoder_type=lidar_encoder_type,
                num_beams=lidar_num_beams,
                output_dim=lidar_embed_dim,
                **lidar_kwargs,
            )
        else:
            self.lidar_encoder = None

        # Critic input: attention output (512) + LiDAR embedding
        critic_input_dim = embedding_dim * 2 + self.lidar_embed_dim

        # Q1 network
        self.layer_1 = nn.Linear(critic_input_dim, 400)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        # Q2 network
        self.layer_4 = nn.Linear(critic_input_dim, 400)
        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def _split_state(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split augmented state into agent_state and lidar."""
        if self.use_lidar:
            agent_state = obs[..., :RAW_STATE_DIM]
            lidar = obs[..., RAW_STATE_DIM:]
            return agent_state, lidar
        else:
            return obs, None

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple:
        """
        Compute twin Q values with late LiDAR fusion.

        Args:
            obs (Tensor): Augmented state (B, N, state_dim).
                          state_dim = 11 + num_beams if use_lidar, else 11.
            action (Tensor): Actions (B*N, action_dim).

        Returns:
            tuple: (Q1, Q2, entropy, hard_logits, distances, hard_weights)
        """
        # Split augmented state
        agent_state, lidar = self._split_state(obs)

        # Original attention on agent state only
        (
            attn_out,
            hard_logits,
            unnorm_rel_dist,
            mean_entropy,
            hard_weights,
            _,
        ) = self.attention(agent_state)

        # Late fusion with LiDAR
        if self.use_lidar and lidar is not None:
            if lidar.dim() == 3:
                batch_size, n_agents, num_beams = lidar.shape
                lidar_flat = lidar.reshape(-1, num_beams)
            else:
                lidar_flat = lidar

            lidar_embed = self.lidar_encoder(lidar_flat)
            fused = torch.cat([attn_out, lidar_embed], dim=-1)
        else:
            fused = attn_out

        # Q1
        s1 = F.leaky_relu(self.layer_1(fused))
        s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(action))
        q1 = self.layer_3(s1)

        # Q2
        s2 = F.leaky_relu(self.layer_4(fused))
        s2 = F.leaky_relu(self.layer_5_s(s2) + self.layer_5_a(action))
        q2 = self.layer_6(s2)

        return q1, q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights


class TD3WithLiDAR:
    """
    TD3 agent with LiDAR support using late fusion and augmented state.

    State format: [agent_state (11-dim), lidar_scan (num_beams)]
    This allows reusing the original ReplayBuffer.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
        num_robots: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        save_every: int = 0,
        load_model: bool = False,
        save_directory: Path = Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        model_name: str = "marlTD3_lidar",
        load_model_name: Optional[str] = None,
        load_directory: Path = Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        attention: str = "igs",
        # LiDAR configuration
        use_lidar: bool = True,
        lidar_encoder_type: str = "sector",  # "sector", "mlp", or "cnn"
        lidar_num_beams: int = 180,
        lidar_embed_dim: int = 12,  # Small for sector encoder
        lidar_encoder_kwargs: Optional[dict] = None,
        lidar_range_max: float = 7.0,
    ):
        """
        Initialize TD3 with late LiDAR fusion.

        Args:
            state_dim (int): Per-agent state dimension (without LiDAR).
            action_dim (int): Per-agent action dimension.
            max_action (float): Action clip magnitude.
            device (torch.device): Target device.
            num_robots (int): Number of agents.
            lr_actor (float): Actor learning rate.
            lr_critic (float): Critic learning rate.
            save_every (int): Save frequency.
            load_model (bool): Whether to load weights.
            save_directory (Path): Save directory.
            model_name (str): Model name for checkpoints.
            load_model_name (str, optional): Name for loading.
            load_directory (Path): Load directory.
            attention (str): Attention type ("igs" or "g2anet").
            use_lidar (bool): Whether to use LiDAR.
            lidar_encoder_type (str): "sector" (simple), "mlp", or "cnn".
            lidar_num_beams (int): Number of LiDAR beams.
            lidar_embed_dim (int): LiDAR embedding dimension.
            lidar_encoder_kwargs (dict): Extra args for encoder.
            lidar_range_max (float): Max LiDAR range for normalization.
        """
        self.num_robots = num_robots
        self.device = device
        self.use_lidar = use_lidar
        self.lidar_encoder_type = lidar_encoder_type
        self.lidar_num_beams = lidar_num_beams
        self.lidar_embed_dim = lidar_embed_dim
        self.lidar_encoder_kwargs = lidar_encoder_kwargs or {}
        self.lidar_range_max = lidar_range_max
        self.state_dim = state_dim

        # Initialize Actor with late fusion
        self.actor = ActorWithLiDAR(
            action_dim,
            embedding_dim=256,
            attention=attention,
            use_lidar=use_lidar,
            lidar_encoder_type=lidar_encoder_type,
            lidar_num_beams=lidar_num_beams,
            lidar_embed_dim=lidar_embed_dim,
            lidar_encoder_kwargs=lidar_encoder_kwargs,
        ).to(device)

        self.actor_target = ActorWithLiDAR(
            action_dim,
            embedding_dim=256,
            attention=attention,
            use_lidar=use_lidar,
            lidar_encoder_type=lidar_encoder_type,
            lidar_num_beams=lidar_num_beams,
            lidar_embed_dim=lidar_embed_dim,
            lidar_encoder_kwargs=lidar_encoder_kwargs,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize Critic with late fusion
        self.critic = CriticWithLiDAR(
            action_dim,
            embedding_dim=256,
            attention=attention,
            use_lidar=use_lidar,
            lidar_encoder_type=lidar_encoder_type,
            lidar_num_beams=lidar_num_beams,
            lidar_embed_dim=lidar_embed_dim,
            lidar_encoder_kwargs=lidar_encoder_kwargs,
        ).to(device)

        self.critic_target = CriticWithLiDAR(
            action_dim,
            embedding_dim=256,
            attention=attention,
            use_lidar=use_lidar,
            lidar_encoder_type=lidar_encoder_type,
            lidar_num_beams=lidar_num_beams,
            lidar_embed_dim=lidar_embed_dim,
            lidar_encoder_kwargs=lidar_encoder_kwargs,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.action_dim = action_dim
        self.max_action = max_action
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0

        if load_model_name is None:
            load_model_name = model_name
        if load_model:
            self.load(filename=load_model_name, directory=load_directory)

        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

    def get_action(
        self,
        obs: np.ndarray,
        add_noise: bool,
    ) -> Tuple:
        """
        Compute action with optional exploration noise.

        Args:
            obs (np.ndarray): Augmented observations (N, state_dim + num_beams).
                              If use_lidar=False, just (N, state_dim).
            add_noise (bool): Whether to add exploration noise.

        Returns:
            tuple: (actions, connection_logits, combined_weights)
        """
        action, connection, combined_weights = self.act(obs)
        if add_noise:
            noise = np.random.normal(0, 0.5, size=action.shape)
            noise = [n / 4 if i % 2 else n for i, n in enumerate(noise)]
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action.reshape(-1, 2), connection, combined_weights

    def act(self, obs: np.ndarray) -> Tuple:
        """
        Compute deterministic action from augmented state.

        Args:
            obs (np.ndarray): Augmented state (N, state_dim + num_beams).
                              If use_lidar=False, just (N, state_dim).

        Returns:
            tuple: (action, connection_logits, combined_weights)
        """
        obs_tensor = torch.Tensor(obs).to(self.device)
        action, connection, _, _, _, combined_weights = self.actor(obs_tensor)
        return action.cpu().data.numpy().flatten(), connection, combined_weights

    def train(
        self,
        replay_buffer,
        iterations: int,
        batch_size: int,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        bce_weight: float = 0.01,
        entropy_weight: float = 1.0,
        connection_proximity_threshold: float = 4.0,
        max_grad_norm: float = 7.0,
    ):
        """
        Run TD3 training loop using augmented state.

        Args:
            replay_buffer: Original ReplayBuffer with sample_batch method.
                           Stores augmented state (state + lidar concatenated).
            iterations (int): Number of gradient steps.
            batch_size (int): Batch size.
            discount (float): Discount factor.
            tau (float): Target update rate.
            policy_noise (float): Target policy noise std.
            noise_clip (float): Target noise clip range.
            policy_freq (int): Actor update frequency.
            bce_weight (float): Hard attention BCE loss weight.
            entropy_weight (float): Attention entropy weight.
            connection_proximity_threshold (float): Distance threshold.
            max_grad_norm (float): Gradient clip norm.
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_critic_loss = 0

        for it in range(iterations):
            # Sample batch from original ReplayBuffer
            # States are already augmented (state + lidar concatenated)
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            # Convert to tensors - states are already augmented
            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device).reshape(-1, 1)
            done = torch.Tensor(batch_dones).to(self.device).reshape(-1, 1)

            # Get next action from target (augmented state passed directly)
            with torch.no_grad():
                next_action, _, _, _, _, _ = self.actor_target(next_state)
                noise = (torch.randn_like(next_action) * policy_noise).clamp(
                    -noise_clip, noise_clip
                )
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # Get target Q values (augmented state passed directly)
                target_Q1, target_Q2, _, _, _, _ = self.critic_target(
                    next_state, next_action
                )
                target_Q = torch.min(target_Q1, target_Q2)
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q).item())
                target_Q = reward + ((1 - done) * discount * target_Q)

            # Get current Q values (augmented state passed directly)
            current_Q1, current_Q2, entropy, hard_logits, pair_d, _ = self.critic(
                state, action
            )

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            # Add entropy bonus
            critic_loss -= entropy_weight * entropy

            av_critic_loss += critic_loss.item()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

            # Delayed actor update
            if it % policy_freq == 0:
                actor_action, actor_hard_logits, actor_pair_d, actor_entropy, _, _ = (
                    self.actor(state)
                )
                actor_loss, _, _, _, _, _ = self.critic(state, actor_action)
                actor_loss = -actor_loss.mean()

                # Add entropy bonus to actor
                actor_loss -= entropy_weight * actor_entropy

                av_loss += actor_loss.item()

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.actor_optimizer.step()

                # Soft update targets
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

        self.iter_count += 1

        # Log metrics
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        self.writer.add_scalar(
            "train/critic_loss", av_critic_loss / iterations, self.iter_count
        )

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename: str, directory: Path):
        """Save model checkpoints."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(
            self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth"
        )
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(
            self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth"
        )
        print(f"Model saved to: {directory}/{filename}")

    def load(self, filename: str, directory: Path):
        """Load model checkpoints."""
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        self.actor_target.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{directory}/{filename}_critic.pth", map_location=self.device)
        )
        self.critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_critic_target.pth", map_location=self.device)
        )
        print(f"Loaded weights from: {directory}/{filename}")

    def prepare_state(
        self,
        poses: List,
        distance: List,
        cos: List,
        sin: List,
        collision: List,
        action: List,
        goal_positions: List,
        lidar_scans: Optional[List] = None,
    ) -> Tuple:
        """
        Convert raw environment outputs into per-agent augmented state vectors.

        State format:
            - If use_lidar: [agent_state (11-dim), lidar (num_beams)] = augmented state
            - If not use_lidar: [agent_state (11-dim)] only

        Args:
            poses (list): Per-agent poses [[x, y, theta], ...].
            distance (list): Per-agent distances to goal.
            cos (list): Per-agent cos(heading error to goal).
            sin (list): Per-agent sin(heading error to goal).
            collision (list): Per-agent collision flags.
            action (list): Per-agent last actions.
            goal_positions (list): Per-agent goals.
            lidar_scans (list, optional): Per-agent LiDAR scans (N, num_beams).

        Returns:
            tuple: (augmented_states, terminal)
                   augmented_states: list of (11 + num_beams) if use_lidar, else list of (11,)
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

            agent_state = [
                px,
                py,
                heading_cos,
                heading_sin,
                distance[i] / 17,
                cos[i],
                sin[i],
                lin_vel,
                ang_vel,
                gx,
                gy,
            ]

            # Augment with LiDAR if enabled
            if self.use_lidar and lidar_scans is not None:
                lidar = lidar_scans[i]  # (num_beams,)
                # Concatenate agent state with normalized lidar
                augmented = np.concatenate([agent_state, lidar])
                states.append(augmented)
            else:
                states.append(agent_state)

            terminal.append(collision[i])

        return states, terminal
