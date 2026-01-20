from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.Attention.g2anet import G2ANet
from robot_nav.models.MARL.Attention.iga import Attention

class CentralizedActor(nn.Module):
    def __init__(self, joint_action_dim, embedding_dim, attention, number_robots):
        super().__init__()
        if attention == "igs":
            self.attention = Attention(embedding_dim)
        elif attention == "g2anet":
            self.attention = G2ANet(embedding_dim)  # ➊ edge classifier
        else:
            raise ValueError("unknown attention mechanism in Actor")
        
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 * number_robots, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, joint_action_dim),
            nn.Tanh(),
        )
            
    def forward(self, obs, detach_attn=False):
        attn_out, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = (
            self.attention(obs)
        )
        if detach_attn:  # used in the policy phase
            attn_out = attn_out.detach()
        batch_size = obs.shape[0] if obs.dim() == 3 else 1
        attn_out = attn_out.view(batch_size, -1)  # flatten all robots' features
        
        action = self.policy_head(attn_out)
        return action, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights
        
class CentralizedCritic(nn.Module):
    def __init__(self, joint_action_dim, embedding_dim, attention, number_robots):
        super().__init__()
        self.embedding_dim = embedding_dim
        if attention == "igs":
            self.attention = Attention(embedding_dim)
        elif attention == "g2anet":
            self.attention = G2ANet(embedding_dim)  # ➊ edge classifier
        else:
            raise ValueError("unknown attention mechanism in Critic")    
        
        self.layer_1 = nn.Linear(embedding_dim * 2 * number_robots, 512)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(512, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(joint_action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(self.embedding_dim * 2 * number_robots, 512)
        torch.nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(512, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(joint_action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")
    
    def forward(self, embedding, joint_action):
            """
            Compute twin Q values from attended embeddings and actions.

            Args:
                embedding (Tensor): Agent embeddings of shape (B, N, state_dim).
                joint_action (Tensor): Actions of shape (B, joint_action_dim), which is (B, N * action_dim).

            Returns:
                tuple:
                    Q1 (Tensor): First Q-value estimate, shape (B, 1).
                    Q2 (Tensor): Second Q-value estimate, shape (B, 1).
                    mean_entropy (Tensor): Mean soft-attention entropy (scalar tensor).
                    hard_logits (Tensor): Hard attention logits, shape (B*N, N-1).
                    unnorm_rel_dist (Tensor): Unnormalized pairwise distances, shape (B*N, N-1, 1).
                    hard_weights (Tensor): Binary hard attention mask, shape (B, N, N).
            """

            (
                embedding_with_attention,
                hard_logits,
                unnorm_rel_dist,
                mean_entropy,
                hard_weights,
                _,
            ) = self.attention(embedding)
            
            # Reshape from (B*N, 512) to (B, N*512)
            batch_size = embedding.shape[0]
            embedding_flat = embedding_with_attention.view(batch_size, -1)

            # Q1
            s1 = F.leaky_relu(self.layer_1(embedding_flat))
            s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(joint_action))  # ✅ No .data
            q1 = self.layer_3(s1)

            # Q2
            s2 = F.leaky_relu(self.layer_4(embedding_flat))
            s2 = F.leaky_relu(self.layer_5_s(s2) + self.layer_5_a(joint_action))  # ✅ No .data
            q2 = self.layer_6(s2)

            return q1, q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights        
        
        
class marlTD3_centralized(object):
    def __init__(
        self,
        state_dim,
        joint_action_dim,
        max_action,
        device,
        num_robots,
        lr_actor=1e-4,
        lr_critic=3e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/MARL/marlTD3_centralized/checkpoint"),
        model_name="marlTD3_centralized",
        load_model_name=None,
        load_directory=Path("robot_nav/models/MARL/marlTD3_centralized/checkpoint"),
        attention="g2anet",
        load_pretrained_attention=False,
        pretrained_attention_model_name=None,
        pretrained_attention_directory=None,
        freeze_attention=False,
    ):
        # Initialize the Actor network
        if attention not in ["igs", "g2anet"]:
            raise ValueError("unknown attention mechanism specified for TD3 model")
        self.num_robots = num_robots
        self.device = device
        self.actor = CentralizedActor(joint_action_dim, embedding_dim=256, attention=attention, number_robots=num_robots).to(
            self.device
        )  # Using the updated Actor
        self.actor_target = CentralizedActor(joint_action_dim, embedding_dim=256, attention=attention, number_robots=num_robots
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.attn_params = list(self.actor.attention.parameters())
        self.policy_params = list(self.actor.policy_head.parameters())

        self.actor_optimizer = torch.optim.Adam(
            self.policy_params + self.attn_params, lr=lr_actor
        )  # TD3 policy

        self.critic = CentralizedCritic(joint_action_dim, embedding_dim=256, attention=attention, number_robots=num_robots).to(
            self.device
        )  # Using the updated Critic
        self.critic_target = CentralizedCritic(
            joint_action_dim, embedding_dim=256, attention=attention, number_robots=num_robots
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(), lr=lr_critic
        )
        self.joint_action_dim = joint_action_dim    
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        if load_model_name is None:
            load_model_name = model_name
        if load_model:
            self.load(filename=load_model_name, directory=load_directory)
        
        # Load pretrained attention weights from decentralized model
        if load_pretrained_attention:
            if pretrained_attention_model_name is None:
                raise ValueError(
                    "pretrained_attention_model_name must be provided when "
                    "load_pretrained_attention=True"
                )
            if pretrained_attention_directory is None:
                raise ValueError(
                    "pretrained_attention_directory must be provided when "
                    "load_pretrained_attention=True"
                )
            self.load_pretrained_attention(
                filename=pretrained_attention_model_name,
                directory=pretrained_attention_directory,
                freeze_attention=freeze_attention,
            )
        
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        
    def get_action(self, obs, add_noise):
        """
        Compute an action for the given observation, with optional exploration noise.

        Args:
            obs (np.ndarray): Observation array of shape (N, state_dim) or (B, N, state_dim).
            add_noise (bool): If True, adds Gaussian exploration noise and clips to bounds.

        Returns:
            tuple:
                action (np.ndarray): joint action reshaped to (N, action_dim=joint_action_dim/N).
                connection_logits (Tensor): Hard attention logits from the actor.
                combined_weights (Tensor): Soft attention weights per (receiver, sender).
        """
        
        action, connection, combined_weights = self.act(obs)
        if add_noise:
            noise = np.random.normal(0, 0.5, size=action.shape)
            noise = [n / 4 if i % 2 else n for i, n in enumerate(noise)]
            action = (action + noise).clip(-self.max_action, self.max_action)
            
        # need to change the reshape according to centralized action format
        return action.reshape(-1, 2), connection, combined_weights
        

    def act(self, state):
        """
        Compute the deterministic action from the current policy.

        Args:
            state (np.ndarray): Observation array of shape (N, state_dim).

        Returns:
            tuple:
                action (np.ndarray): Flattened action vector of shape (joint_action_dim,).
                connection_logits (Tensor): Hard attention logits from the actor.
                combined_weights (Tensor): Soft attention weights per (receiver, sender).
        """ 
        # Function to get the action from the actor
        state = torch.Tensor(state).to(self.device)
        # res = self.attention(state)
        action, connection, _, _, _, combined_weights = self.actor(state)
        return action.cpu().data.numpy().flatten(), connection, combined_weights
    
    # training cycle
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
        bce_weight=0.01,
        entropy_weight=1,
        connection_proximity_threshold=4,
        max_grad_norm=7.0,
    ):
        """
        Run a TD3 training loop over sampled mini-batches.

        Args:
            replay_buffer: Buffer supporting ``sample_batch(batch_size)`` -> tuple of arrays.
            iterations (int): Number of gradient steps.
            batch_size (int): Mini-batch size.
            discount (float, optional): Discount factor γ. Defaults to 0.99.
            tau (float, optional): Target network update rate. Defaults to 0.005.
            policy_noise (float, optional): Std of target policy noise. Defaults to 0.2.
            noise_clip (float, optional): Clipping range for target noise. Defaults to 0.5.
            policy_freq (int, optional): Actor update period (in critic steps). Defaults to 2.
            bce_weight (float, optional): Weight for hard-connection BCE loss. Defaults to 0.01.
            entropy_weight (float, optional): Weight for attention entropy bonus. Defaults to 1.
            connection_proximity_threshold (float, optional): Distance threshold for the
                positive class when supervising hard connections. Defaults to 4.
            max_grad_norm (float, optional): Gradient clipping norm. Defaults to 7.0.

        Returns:
            None
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_critic_loss = 0
        av_critic_entropy = []
        av_actor_entropy = []
        av_actor_loss = 0
        av_critic_bce_loss = []
        av_actor_bce_loss = []

        for it in range(iterations):
            # sample a batch
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            state = (
                torch.Tensor(batch_states)
                .to(self.device)
                .view(batch_size, self.num_robots, self.state_dim)
            )
            next_state = (
                torch.Tensor(batch_next_states)
                .to(self.device)
                .view(batch_size, self.num_robots, self.state_dim)
            )
            action = (
                torch.Tensor(batch_actions)
                .to(self.device)
                .view(batch_size, self.joint_action_dim)
            )
            # Aggregate rewards: average across all robots for centralized Q-value
            # Using mean instead of sum to keep reward scale stable
            reward = (
                torch.Tensor(batch_rewards)
                .to(self.device)
                .view(batch_size, self.num_robots)
                .mean(dim=1, keepdim=True)  # (B, 1) - average reward for joint action
            )
            # Buffer stores [done, done, done, ...] - same value repeated for each robot
            # We just take the first one (all are identical)
            done = (
                torch.Tensor(batch_dones)
                .to(self.device)
                .view(batch_size, self.num_robots)[:, 0:1]  # (B, 1)
            )

            with torch.no_grad():
                next_action, _, _, _, _, _ = self.actor_target(
                    next_state, detach_attn=True
                )

            # --- Target smoothing ---
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            ).view(batch_size, self.joint_action_dim) # Reshapes to decentralized format (B*N, 2)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # --- Target Q values ---
            target_Q1, target_Q2, _, _, _, _ = self.critic_target(
                next_state, next_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += target_Q.mean()
            max_Q = max(max_Q, target_Q.max().item())
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # --- Critic update ---
            (
                current_Q1,
                current_Q2,
                mean_entropy,
                hard_logits,
                unnorm_rel_dist,
                hard_weights,
            ) = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            targets = (
                unnorm_rel_dist.flatten() < connection_proximity_threshold
            ).float()
            flat_logits = hard_logits.flatten()
            bce_loss = F.binary_cross_entropy_with_logits(flat_logits, targets)

            av_critic_bce_loss.append(bce_loss)

            total_loss = (
                critic_loss - entropy_weight * mean_entropy + bce_weight * bce_loss
            )
            av_critic_entropy.append(mean_entropy)

            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

            av_loss += total_loss.item()
            av_critic_loss += critic_loss.item()

            # --- Actor update ---
            if it % policy_freq == 0:

                action, hard_logits, unnorm_rel_dist, mean_entropy, hard_weights, _ = (
                    self.actor(state, detach_attn=False)
                )
                targets = (
                    unnorm_rel_dist.flatten() < connection_proximity_threshold
                ).float()
                flat_logits = hard_logits.flatten()
                bce_loss = F.binary_cross_entropy_with_logits(flat_logits, targets)

                av_actor_bce_loss.append(bce_loss)

                actor_Q, _, _, _, _, _ = self.critic(state, action)
                actor_loss = -actor_Q.mean()
                total_loss = (
                    actor_loss - entropy_weight * mean_entropy + bce_weight * bce_loss
                )
                av_actor_entropy.append(mean_entropy)

                self.actor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_params, max_grad_norm)
                self.actor_optimizer.step()

                av_actor_loss += total_loss.item()

                # Soft update target networks
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
        self.writer.add_scalar(
            "train/loss_total", av_loss / iterations, self.iter_count
        )
        self.writer.add_scalar(
            "train/critic_loss", av_critic_loss / iterations, self.iter_count
        )
        self.writer.add_scalar(
            "train/av_critic_entropy",
            sum(av_critic_entropy) / len(av_critic_entropy),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_entropy",
            sum(av_actor_entropy) / len(av_actor_entropy),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_critic_bce_loss",
            sum(av_critic_bce_loss) / len(av_critic_bce_loss),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_bce_loss",
            sum(av_actor_bce_loss) / len(av_actor_bce_loss),
            self.iter_count,
        )
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)

        self.writer.add_scalar(
            "train/actor_loss",
            av_actor_loss / (iterations // policy_freq),
            self.iter_count,
        )

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory, epoch=None):
        """
        Saves the current model parameters to the specified directory.

        Args:
            filename (str): Base filename for saved files.
            directory (Path): Path to save the model files.
            epoch (int, optional): If provided, appends epoch number to filename
                for versioned checkpoints (e.g., "model_epoch_2000_actor.pth").
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create versioned filename if epoch is provided
        if epoch is not None:
            save_name = f"{filename}_epoch_{epoch}"
        else:
            save_name = filename
            
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, save_name))
        torch.save(
            self.actor_target.state_dict(),
            "%s/%s_actor_target.pth" % (directory, save_name),
        )
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, save_name))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, save_name),
        )
        
        if epoch is not None:
            print(f"Saved checkpoint at epoch {epoch} to: {directory}/{save_name}")

    def save_checkpoint(self, epoch):
        """
        Save a versioned checkpoint with the epoch number in the filename.
        
        This creates separate checkpoint files that won't be overwritten,
        useful for saving milestones during training.
        
        Args:
            epoch (int): Current epoch number to include in filename.
        """
        self.save(filename=self.model_name, directory=self.save_directory, epoch=epoch)

    def load(self, filename, directory):
        """
        Loads model parameters from the specified directory.

        Args:
            filename (str): Base filename for saved files.
            directory (Path): Path to load the model files from.
        """
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.actor_target.load_state_dict(
            torch.load("%s/%s_actor_target.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename))
        )
        print(f"Loaded weights from: {directory}")

    def load_pretrained_attention(self, filename, directory, freeze_attention=False):
        """
        Load pretrained attention network weights from a decentralized MARL model.

        This method loads only the attention module weights from a pretrained
        decentralized marlTD3 model into the centralized actor and critic networks.
        The policy head and critic Q-network layers are left with their current
        (randomly initialized) weights.

        Args:
            filename (str): Base filename for the pretrained model files
                (e.g., "TDR-MARL-train").
            directory (Path or str): Path to the directory containing the
                pretrained model files.
            freeze_attention (bool, optional): If True, freezes the attention
                network parameters so they are not updated during training.
                Defaults to False.

        Example:
            >>> model.load_pretrained_attention(
            ...     filename="TDR-MARL-train",
            ...     directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
            ...     freeze_attention=True
            ... )
        """
        # Load pretrained actor state dict (from decentralized model)
        pretrained_actor_path = "%s/%s_actor.pth" % (directory, filename)
        pretrained_actor_state = torch.load(
            pretrained_actor_path, map_location=self.device
        )

        # Load pretrained critic state dict (from decentralized model)
        pretrained_critic_path = "%s/%s_critic.pth" % (directory, filename)
        pretrained_critic_state = torch.load(
            pretrained_critic_path, map_location=self.device
        )

        # Extract only attention-related keys from actor
        attention_keys_actor = {
            k: v for k, v in pretrained_actor_state.items() if k.startswith("attention.")
        }

        # Extract only attention-related keys from critic
        attention_keys_critic = {
            k: v for k, v in pretrained_critic_state.items() if k.startswith("attention.")
        }

        # Load attention weights into actor
        actor_state = self.actor.state_dict()
        actor_state.update(attention_keys_actor)
        self.actor.load_state_dict(actor_state)

        # Load attention weights into actor_target
        actor_target_state = self.actor_target.state_dict()
        actor_target_state.update(attention_keys_actor)
        self.actor_target.load_state_dict(actor_target_state)

        # Load attention weights into critic
        critic_state = self.critic.state_dict()
        critic_state.update(attention_keys_critic)
        self.critic.load_state_dict(critic_state)

        # Load attention weights into critic_target
        critic_target_state = self.critic_target.state_dict()
        critic_target_state.update(attention_keys_critic)
        self.critic_target.load_state_dict(critic_target_state)

        print(f"Loaded pretrained attention weights from: {directory}/{filename}")
        print(f"  Actor attention keys loaded: {len(attention_keys_actor)}")
        print(f"  Critic attention keys loaded: {len(attention_keys_critic)}")

        # Optionally freeze attention parameters
        if freeze_attention:
            self._freeze_attention_parameters()
            print("  Attention parameters frozen (will not be updated during training)")

    def _freeze_attention_parameters(self):
        """
        Freeze attention network parameters to prevent updates during training.

        This is useful when you want to use pretrained attention weights and
        only train the policy head and critic Q-networks.
        """
        for param in self.actor.attention.parameters():
            param.requires_grad = False
        for param in self.actor_target.attention.parameters():
            param.requires_grad = False
        for param in self.critic.attention.parameters():
            param.requires_grad = False
        for param in self.critic_target.attention.parameters():
            param.requires_grad = False

        # Update optimizer to only include trainable parameters
        self.attn_params = []  # Clear attention params from optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.policy_params, lr=self.actor_optimizer.defaults['lr']
        )

    def unfreeze_attention_parameters(self):
        """
        Unfreeze attention network parameters to allow updates during training.

        Call this after initially training with frozen attention if you want to
        fine-tune the full model.
        """
        for param in self.actor.attention.parameters():
            param.requires_grad = True
        for param in self.actor_target.attention.parameters():
            param.requires_grad = True
        for param in self.critic.attention.parameters():
            param.requires_grad = True
        for param in self.critic_target.attention.parameters():
            param.requires_grad = True

        # Update optimizer to include attention parameters again
        self.attn_params = list(self.actor.attention.parameters())
        self.actor_optimizer = torch.optim.Adam(
            self.policy_params + self.attn_params,
            lr=self.actor_optimizer.defaults['lr']
        )
        print("Attention parameters unfrozen (will be updated during training)")

    def prepare_state(
        self, poses, distance, cos, sin, collision, action, goal_positions
    ):
        """
        Convert raw environment outputs into per-agent state vectors.

        Args:
            poses (list): Per-agent poses [[x, y, theta], ...].
            distance (list): Per-agent distances to goal.
            cos (list): Per-agent cos(heading error to goal).
            sin (list): Per-agent sin(heading error to goal).
            collision (list): Per-agent collision flags (bool or {0,1}).
            action (list): Per-agent last actions [[lin_vel, ang_vel], ...].
            goal_positions (list): Per-agent goals [[gx, gy], ...].

        Returns:
            tuple:
                states (list): Per-agent state vectors (length == state_dim).
                terminal (list): Terminal flags (collision or goal), same length as states.
        """
        states = []
        terminal = []

        for i in range(self.num_robots):
            pose = poses[i]  # [x, y, theta]
            goal_pos = goal_positions[i]  # [goal_x, goal_y]
            act = action[i]  # [lin_vel, ang_vel]

            px, py, theta = pose
            gx, gy = goal_pos

            # Heading as cos/sin
            heading_cos = np.cos(theta)
            heading_sin = np.sin(theta)

            # Last velocity
            lin_vel = act[0] * 2  # Assuming original range [0, 0.5]
            ang_vel = (act[1] + 1) / 2  # Assuming original range [-1, 1]

            # Final state vector
            state = [
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

            assert (
                len(state) == self.state_dim
            ), f"State length mismatch: expected {self.state_dim}, got {len(state)}"
            states.append(state)
            # terminal.append(collision[i])

        return states