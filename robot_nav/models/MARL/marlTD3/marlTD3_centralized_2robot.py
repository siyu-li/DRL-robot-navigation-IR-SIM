from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.Attention.g2anet import G2ANet
from robot_nav.models.MARL.Attention.iga import Attention


class TwoRobotCentralizedActor(nn.Module):
    """
    Centralized actor for 2 active robots using attention from full N-robot graph.
    
    The attention network processes all N robots but the policy head only outputs
    actions for the 2 active robots.
    """
    
    def __init__(self, embedding_dim, attention_type, num_total_robots, active_robot_ids, coupled_mode=False):
        super().__init__()
        self.num_total_robots = num_total_robots
        self.active_robot_ids = active_robot_ids
        self.coupled_mode = coupled_mode
        self.embedding_dim = embedding_dim
        
        # Attention network for full graph (pretrained)
        if attention_type == "igs":
            self.attention = Attention(embedding_dim)
        elif attention_type == "g2anet":
            self.attention = G2ANet(embedding_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Policy head for 2 robots only
        # Input: 2 * (embedding_dim * 2) = 2 * 512 = 1024
        # Output: 4 (lin_vel_1, ang_vel_1, lin_vel_2, ang_vel_2) or 3 if coupled
        output_dim = 3 if coupled_mode else 4
        
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        )
    
    def forward(self, obs, detach_attn=False):
        """
        Forward pass through attention and policy head.
        
        Args:
            obs: Observations for ALL N robots, shape (B, N, state_dim) or (N, state_dim).
            detach_attn: If True, detach attention output before policy head.
            
        Returns:
            action: Actions for 2 active robots.
            hard_logits, pair_d, mean_entropy, hard_weights, combined_weights: Attention diagnostics.
        """
        # Run attention on full graph
        attn_out, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = (
            self.attention(obs)
        )
        
        if detach_attn:
            attn_out = attn_out.detach()
        
        # Extract embeddings for active robots only
        batch_size = obs.shape[0] if obs.dim() == 3 else 1
        attn_out = attn_out.view(batch_size, self.num_total_robots, -1)  # (B, N, 512)
        
        # Select only active robot embeddings
        active_embeddings = attn_out[:, self.active_robot_ids, :]  # (B, 2, 512)
        active_embeddings = active_embeddings.view(batch_size, -1)  # (B, 1024)
        
        action = self.policy_head(active_embeddings)  # (B, 4) or (B, 3)
        
        return action, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights


class TwoRobotCentralizedCritic(nn.Module):
    """
    Centralized critic for 2 active robots.
    
    Evaluates joint Q-value based on full attention graph but only active robot actions.
    """
    
    def __init__(self, embedding_dim, attention_type, num_total_robots, active_robot_ids, coupled_mode=False):
        super().__init__()
        self.num_total_robots = num_total_robots
        self.active_robot_ids = active_robot_ids
        self.coupled_mode = coupled_mode
        self.embedding_dim = embedding_dim
        
        if attention_type == "igs":
            self.attention = Attention(embedding_dim)
        elif attention_type == "g2anet":
            self.attention = G2ANet(embedding_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        action_dim = 3 if coupled_mode else 4
        
        # Twin Q-networks for 2 active robots
        self.q1_layer1 = nn.Linear(embedding_dim * 2 * 2, 256)
        self.q1_layer2_s = nn.Linear(256, 128)
        self.q1_layer2_a = nn.Linear(action_dim, 128)
        self.q1_layer3 = nn.Linear(128, 1)
        
        self.q2_layer1 = nn.Linear(embedding_dim * 2 * 2, 256)
        self.q2_layer2_s = nn.Linear(256, 128)
        self.q2_layer2_a = nn.Linear(action_dim, 128)
        self.q2_layer3 = nn.Linear(128, 1)
        
        # Initialize weights
        for layer in [self.q1_layer1, self.q1_layer2_s, self.q1_layer2_a, self.q1_layer3,
                      self.q2_layer1, self.q2_layer2_s, self.q2_layer2_a, self.q2_layer3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
    
    def forward(self, obs, action):
        """
        Compute twin Q-values.
        
        Args:
            obs: Observations for ALL N robots, shape (B, N, state_dim).
            action: Joint action for 2 active robots, shape (B, 4) or (B, 3).
            
        Returns:
            q1, q2: Q-value estimates.
            mean_entropy, hard_logits, unnorm_rel_dist, hard_weights: Attention diagnostics.
        """
        attn_out, hard_logits, unnorm_rel_dist, mean_entropy, hard_weights, _ = (
            self.attention(obs)
        )
        
        batch_size = obs.shape[0] if obs.dim() == 3 else 1
        attn_out = attn_out.view(batch_size, self.num_total_robots, -1)
        
        # Select active robot embeddings
        active_embeddings = attn_out[:, self.active_robot_ids, :]
        active_embeddings = active_embeddings.view(batch_size, -1)
        
        # Q1
        s1 = F.leaky_relu(self.q1_layer1(active_embeddings))
        s1 = F.leaky_relu(self.q1_layer2_s(s1) + self.q1_layer2_a(action))
        q1 = self.q1_layer3(s1)
        
        # Q2
        s2 = F.leaky_relu(self.q2_layer1(active_embeddings))
        s2 = F.leaky_relu(self.q2_layer2_s(s2) + self.q2_layer2_a(action))
        q2 = self.q2_layer3(s2)
        
        return q1, q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights


class marlTD3_2robot(object):
    """
    TD3 agent for centralized control of 2 robots within an N-robot environment.
    
    Uses pretrained attention networks from decentralized training.
    """
    
    def __init__(
        self,
        state_dim,
        num_total_robots,
        active_robot_ids,
        device,
        coupled_mode=False,
        lr_actor=1e-4,
        lr_critic=3e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/MARL/marlTD3_centralized_2robot/checkpoint"),
        model_name="marlTD3_2robot",
        load_model_name=None,
        load_directory=None,
        attention="igs",
        load_pretrained_attention=False,
        pretrained_attention_model_name=None,
        pretrained_attention_directory=None,
        freeze_attention=True,
    ):
        self.device = device
        self.num_total_robots = num_total_robots
        self.num_active_robots = 2
        self.active_robot_ids = active_robot_ids
        self.coupled_mode = coupled_mode
        self.state_dim = state_dim
        self.action_dim = 3 if coupled_mode else 4
        self.max_action = 1.0
        
        embedding_dim = 256
        
        # Actor networks
        self.actor = TwoRobotCentralizedActor(
            embedding_dim, attention, num_total_robots, active_robot_ids, coupled_mode
        ).to(device)
        self.actor_target = TwoRobotCentralizedActor(
            embedding_dim, attention, num_total_robots, active_robot_ids, coupled_mode
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = TwoRobotCentralizedCritic(
            embedding_dim, attention, num_total_robots, active_robot_ids, coupled_mode
        ).to(device)
        self.critic_target = TwoRobotCentralizedCritic(
            embedding_dim, attention, num_total_robots, active_robot_ids, coupled_mode
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Separate attention and policy parameters
        self.attn_params = list(self.actor.attention.parameters())
        self.policy_params = list(self.actor.policy_head.parameters())
        
        self.actor_optimizer = torch.optim.Adam(
            self.policy_params + self.attn_params, lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )
        
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        
        # Load pretrained attention
        if load_pretrained_attention:
            self.load_pretrained_attention(
                pretrained_attention_model_name,
                pretrained_attention_directory,
                freeze_attention
            )
        
        if load_model:
            load_name = load_model_name or model_name
            load_dir = load_directory or save_directory
            self.load(load_name, load_dir)

    def load_pretrained_attention(self, filename, directory, freeze=True):
        """Load attention weights from pretrained decentralized model."""
        actor_path = f"{directory}/{filename}_actor.pth"
        critic_path = f"{directory}/{filename}_critic.pth"
        
        pretrained_actor = torch.load(actor_path, map_location=self.device)
        pretrained_critic = torch.load(critic_path, map_location=self.device)
        
        # Extract attention keys
        attn_keys_actor = {k: v for k, v in pretrained_actor.items() if k.startswith("attention.")}
        attn_keys_critic = {k: v for k, v in pretrained_critic.items() if k.startswith("attention.")}
        
        # Load into actor
        actor_state = self.actor.state_dict()
        actor_state.update(attn_keys_actor)
        self.actor.load_state_dict(actor_state)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Load into critic
        critic_state = self.critic.state_dict()
        critic_state.update(attn_keys_critic)
        self.critic.load_state_dict(critic_state)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        print(f"Loaded pretrained attention from {directory}/{filename}")
        
        if freeze:
            self._freeze_attention()
    
    def _freeze_attention(self):
        """Freeze attention parameters."""
        for param in self.actor.attention.parameters():
            param.requires_grad = False
        for param in self.actor_target.attention.parameters():
            param.requires_grad = False
        for param in self.critic.attention.parameters():
            param.requires_grad = False
        for param in self.critic_target.attention.parameters():
            param.requires_grad = False
        
        self.attn_params = []
        self.actor_optimizer = torch.optim.Adam(
            self.policy_params, lr=self.actor_optimizer.defaults['lr']
        )
        print("Attention parameters frozen")

    def get_action(self, obs, add_noise):
        """
        Get action for 2 active robots.
        
        Args:
            obs: Full N-robot observations, shape (N, state_dim).
            add_noise: Whether to add exploration noise.
            
        Returns:
            action: List of 2 actions [[lin, ang], [lin, ang]] or [lin, ang1, ang2] if coupled.
            connection: Attention connection logits.
            combined_weights: Attention weights.
        """
        action, connection, combined_weights = self.act(obs)
        
        if add_noise:
            noise = np.random.normal(0, 0.3, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        if self.coupled_mode:
            # Return as [lin_vel, ang_vel_1, ang_vel_2]
            return action.tolist(), connection, combined_weights
        else:
            # Return as [[lin_1, ang_1], [lin_2, ang_2]]
            return action.reshape(2, 2).tolist(), connection, combined_weights
    
    def act(self, state):
        """Compute action from actor."""
        state = torch.Tensor(state).unsqueeze(0).to(self.device)  # (1, N, state_dim)
        action, connection, _, _, _, combined_weights = self.actor(state)
        return action.cpu().data.numpy().flatten(), connection, combined_weights

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        """Standard TD3 training loop."""
        av_critic_loss = 0
        av_actor_loss = 0
        
        for it in range(iterations):
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            
            # States are for all N robots
            state = torch.Tensor(batch_states).to(self.device).view(
                batch_size, self.num_total_robots, self.state_dim
            )
            next_state = torch.Tensor(batch_next_states).to(self.device).view(
                batch_size, self.num_total_robots, self.state_dim
            )
            
            # Actions are only for 2 active robots
            action = torch.Tensor(batch_actions).to(self.device).view(
                batch_size, self.action_dim
            )
            
            # Rewards are summed for active robots
            reward = torch.Tensor(batch_rewards).to(self.device).view(batch_size, 1)
            done = torch.Tensor(batch_dones).to(self.device).view(batch_size, 1)
            
            with torch.no_grad():
                next_action, _, _, _, _, _ = self.actor_target(next_state, detach_attn=True)
                noise = torch.randn_like(next_action) * policy_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
                target_Q1, target_Q2, _, _, _, _ = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * discount * target_Q
            
            current_Q1, current_Q2, _, _, _, _ = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
            self.critic_optimizer.step()
            
            av_critic_loss += critic_loss.item()
            
            if it % policy_freq == 0:
                actor_action, _, _, _, _, _ = self.actor(state, detach_attn=False)
                actor_Q, _, _, _, _, _ = self.critic(state, actor_action)
                actor_loss = -actor_Q.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_params, 5.0)
                self.actor_optimizer.step()
                
                av_actor_loss += actor_loss.item()
                
                # Soft update targets
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.iter_count += 1
        self.writer.add_scalar("train/critic_loss", av_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/actor_loss", av_actor_loss / (iterations // policy_freq), self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(self.model_name, self.save_directory)

    def prepare_state(self, all_poses, active_distances, active_cos, active_sin, 
                      active_collision, active_action, active_goal_positions):
        """
        Prepare state vectors for ALL robots (needed for attention).
        
        For static robots, use their poses but zero velocities and dummy goals.
        """
        states = []
        
        for i in range(self.num_total_robots):
            pose = all_poses[i]
            px, py, theta = pose
            
            if i in self.active_robot_ids:
                # Active robot: use actual data
                idx = self.active_robot_ids.index(i)
                gx, gy = active_goal_positions[idx]
                dist = active_distances[idx] / 17
                cos_val = active_cos[idx]
                sin_val = active_sin[idx]
                act = active_action[idx] if isinstance(active_action[0], list) else active_action
                lin_vel = act[0] * 2 if isinstance(act, list) else act[idx * 2] * 2
                ang_vel = (act[1] + 1) / 2 if isinstance(act, list) else (act[idx * 2 + 1] + 1) / 2
            else:
                # Static robot: use position but dummy goal/velocity
                gx, gy = px + 1, py + 1  # Dummy goal
                dist = 0.1
                cos_val, sin_val = 1.0, 0.0
                lin_vel, ang_vel = 0.0, 0.5
            
            state = [
                px, py,
                np.cos(theta), np.sin(theta),
                dist, cos_val, sin_val,
                lin_vel, ang_vel,
                gx, gy,
            ]
            states.append(state)
        
        return states

    def save(self, filename, directory):
        """Save model checkpoints."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), directory / f"{filename}_critic.pth")
        print(f"Saved model to {directory}/{filename}")

    def load(self, filename, directory):
        """Load model checkpoints."""
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(
            torch.load(f"{directory}/{filename}_critic.pth", map_location=self.device)
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"Loaded model from {directory}/{filename}")