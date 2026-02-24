"""
PPO Training Framework for Categorical Group Switcher.

Trains a GroupSwitcher policy using Proximal Policy Optimization (PPO)
with a discrete (Categorical) action space. Each "action" is selecting
one of M candidate groups at a selection interval.

Architecture:
    Actor (policy):  Two-tower fusion (same as GroupSwitcher) → per-group logits
                     → Categorical distribution over M groups.
    Critic (value):  Separate state-only tower → V(s) scalar.

Key differences from the continuous PPO in robot_nav/models/PPO/PPO.py:
    - Discrete action space (Categorical) instead of continuous (MultivariateNormal).
    - Variable-size action space: M groups may differ across episodes (though
      typically fixed for a given robot count).
    - Actor input is per-group feature matrix X ∈ R^(M, D); critic input is
      state-level features only (h_glob + global scalars).
    - No action_std decay; exploration is controlled by entropy bonus weight
      and optional temperature annealing.

Usage:
    from robot_nav.models.MARL.switcher.switcher_ppo import SwitcherPPO

    ppo = SwitcherPPO(
        embed_dim=512,
        group_scalar_dim=13,
        state_scalar_dim=5,
        device="cuda",
    )

    # During rollout:
    group_idx = ppo.get_action(group_features_X, state_features, explore=True)

    # After collecting a batch of transitions:
    ppo.train(replay_buffer=None, iterations=10, batch_size=0)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# Rollout Buffer
# =============================================================================
class SwitcherRolloutBuffer:
    """
    Buffer to store rollout transitions for PPO training of the group switcher.

    Each entry corresponds to one group-selection decision (every selection_interval
    simulation steps), not every simulation step.

    Stored fields per decision step:
        group_features:  (M, D) tensor — per-group feature matrix.
        state_features:  (S,) tensor   — state-level features for value head.
        action:          int            — selected group index.
        logprob:         scalar tensor  — log π(action | state).
        value:           scalar tensor  — V(state) estimate.
        reward:          float          — accumulated reward over the interval.
        is_terminal:     bool           — whether episode ended during this interval.
    """

    def __init__(self):
        self.group_features: List[torch.Tensor] = []
        self.state_features: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.is_terminals: List[bool] = []

    def add(
        self,
        group_features: torch.Tensor,
        state_features: torch.Tensor,
        action: int,
        logprob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        is_terminal: bool,
    ):
        """Append one decision-step transition."""
        self.group_features.append(group_features)
        self.state_features.append(state_features)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear(self):
        """Clear all stored transitions."""
        self.group_features.clear()
        self.state_features.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.is_terminals.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# =============================================================================
# Actor-Critic Network
# =============================================================================
class SwitcherActorCritic(nn.Module):
    """
    Actor-Critic network for categorical group selection via PPO.

    Actor (policy head):
        Two-tower fusion that scores each candidate group.
        Input:  X ∈ R^(M, D)  where D = 2*embed_dim + group_scalar_dim
                The embedding portion [h_g ∥ h_glob] is 2*embed_dim = 1024,
                compressed via Linear(1024 → embed_hidden).
        Output: logits ∈ R^(M,)  → Categorical distribution

    Critic (value head):
        Separate state-only tower that estimates V(s).
        Input:  [h_glob (embed_dim) || state_scalars (state_scalar_dim)]
        Output: V(s) scalar

    Args:
        embed_dim: Dimension of per-robot embeddings from GAT backbone output
            (d = 2 * GAT embedding_dim = 512).  ``h_g`` and ``h_glob`` each
            have this dim, so the concatenated embedding input to the actor
            is ``2 * embed_dim = 1024``.
        group_scalar_dim: Number of per-group scalar features
            (size_feat + coupling_mode + attn_stats + extras).
        state_scalar_dim: Number of state-level scalar features for the value head
            (mean_dist, var_dist, min_clearance, frac_reached, steps_frac).
        embed_hidden: Hidden dim for the actor's embedding tower
            (Linear(2*embed_dim → embed_hidden)).
        group_scalar_hidden: Hidden dim for the actor's scalar tower.
        fusion_hidden: Hidden dim for the actor's fusion layer.
        value_embed_hidden: Hidden dim for the critic's embedding tower
            (Linear(embed_dim → value_embed_hidden)).
        value_scalar_hidden: Hidden dim for the critic's scalar tower.
        dropout: Dropout probability in the actor fusion layer.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        group_scalar_dim: int = 13,
        state_scalar_dim: int = 5,
        embed_hidden: int = 256,
        group_scalar_hidden: int = 64,
        fusion_hidden: int = 256,
        value_embed_hidden: int = 128,
        value_scalar_hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.group_scalar_dim = group_scalar_dim
        self.state_scalar_dim = state_scalar_dim

        # ---- Actor (policy): per-group scorer ----
        # Tower 1: embedding tower  [h_g || h_glob] → embed_hidden
        self.actor_embed_tower = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_hidden),
            nn.GELU(),
            nn.LayerNorm(embed_hidden),
        )

        # Tower 2: scalar tower  group_scalars → group_scalar_hidden
        self.actor_scalar_tower = nn.Sequential(
            nn.Linear(group_scalar_dim, group_scalar_hidden),
            nn.GELU(),
            nn.LayerNorm(group_scalar_hidden),
        )

        # Fusion → logit
        actor_fusion_in = embed_hidden + group_scalar_hidden
        self.actor_fusion = nn.Sequential(
            nn.Linear(actor_fusion_in, fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

        # ---- Critic (value): state-only ----
        # Tower 1: global embedding → value_embed_hidden
        self.critic_embed_tower = nn.Sequential(
            nn.Linear(embed_dim, value_embed_hidden),
            nn.GELU(),
            nn.LayerNorm(value_embed_hidden),
        )

        # Tower 2: state scalars → value_scalar_hidden
        self.critic_scalar_tower = nn.Sequential(
            nn.Linear(state_scalar_dim, value_scalar_hidden),
            nn.GELU(),
            nn.LayerNorm(value_scalar_hidden),
        )

        # Fusion → V(s)
        critic_fusion_in = value_embed_hidden + value_scalar_hidden
        self.critic_fusion = nn.Sequential(
            nn.Linear(critic_fusion_in, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for GELU activations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _actor_logits(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute per-group logits from group feature matrix.

        Args:
            X: Group features of shape (M, D), D = 2*embed_dim + group_scalar_dim.

        Returns:
            logits: Shape (M,).
        """
        embed_features = X[:, : 2 * self.embed_dim]
        scalar_features = X[:, 2 * self.embed_dim :]

        e = self.actor_embed_tower(embed_features)   # (M, embed_hidden)
        s = self.actor_scalar_tower(scalar_features)  # (M, group_scalar_hidden)
        fused = torch.cat([e, s], dim=-1)
        logits = self.actor_fusion(fused).squeeze(-1)  # (M,)
        return logits

    def _critic_value(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Compute state value V(s).

        Args:
            state_features: Shape (S,) where S = embed_dim + state_scalar_dim.
                Layout: [h_glob (embed_dim), state_scalars (state_scalar_dim)].

        Returns:
            value: Scalar tensor.
        """
        h_glob = state_features[: self.embed_dim]          # (embed_dim,)
        scalars = state_features[self.embed_dim :]          # (state_scalar_dim,)

        e = self.critic_embed_tower(h_glob.unsqueeze(0))    # (1, value_embed_hidden)
        s = self.critic_scalar_tower(scalars.unsqueeze(0))  # (1, value_scalar_hidden)
        fused = torch.cat([e, s], dim=-1)
        value = self.critic_fusion(fused).squeeze()         # scalar
        return value

    def forward(self):
        """Not used directly. Call act() or evaluate() instead."""
        raise NotImplementedError

    def act(
        self,
        group_features: torch.Tensor,
        state_features: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select a group action, compute log_prob and value.

        Args:
            group_features: (M, D) group feature matrix.
            state_features: (S,) state-level feature vector.
            sample: If True, sample from Categorical; else use argmax.

        Returns:
            action: Selected group index (int).
            log_prob: Log probability of the selected action (detached).
            value: State value estimate (detached).
        """
        logits = self._actor_logits(group_features)
        dist = Categorical(logits=logits)

        if sample:
            action = dist.sample()
        else:
            action = logits.argmax()

        log_prob = dist.log_prob(action)
        value = self._critic_value(state_features)

        return action.item(), log_prob.detach(), value.detach()

    def evaluate(
        self,
        group_features_list: List[torch.Tensor],
        state_features_batch: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored transitions for PPO update.

        Uses batched computation when all timesteps have the same number of
        groups M (typical case), falling back to per-sample processing otherwise.

        Args:
            group_features_list: List of T tensors, each (M_t, D).
            state_features_batch: (T, S) state features.
            actions: (T,) long tensor of selected group indices.

        Returns:
            log_probs: (T,) log probabilities under current policy.
            values: (T,) state value estimates.
            entropies: (T,) policy entropies.
        """
        T = len(group_features_list)

        # Check if all group feature tensors have the same M (typical case)
        M0 = group_features_list[0].shape[0]
        all_same_M = all(x.shape[0] == M0 for x in group_features_list)

        if all_same_M:
            # ---- Batched path: stack into (T, M, D) and process together ----
            # Actor: batch all group features
            X_batch = torch.stack(group_features_list)  # (T, M, D)
            T_b, M_b, D_b = X_batch.shape
            X_flat = X_batch.view(T_b * M_b, D_b)      # (T*M, D)

            embed_features = X_flat[:, : 2 * self.embed_dim]
            scalar_features = X_flat[:, 2 * self.embed_dim :]

            e = self.actor_embed_tower(embed_features)    # (T*M, embed_hidden)
            s = self.actor_scalar_tower(scalar_features)  # (T*M, group_scalar_hidden)
            fused = torch.cat([e, s], dim=-1)
            logits_flat = self.actor_fusion(fused).squeeze(-1)  # (T*M,)
            logits = logits_flat.view(T_b, M_b)           # (T, M)

            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)             # (T,)
            entropies = dist.entropy()                     # (T,)

            # Critic: already batched via state_features_batch (T, S)
            h_glob_batch = state_features_batch[:, : self.embed_dim]      # (T, embed_dim)
            scalars_batch = state_features_batch[:, self.embed_dim :]      # (T, state_scalar_dim)

            e_v = self.critic_embed_tower(h_glob_batch)    # (T, value_embed_hidden)
            s_v = self.critic_scalar_tower(scalars_batch)  # (T, value_scalar_hidden)
            fused_v = torch.cat([e_v, s_v], dim=-1)
            values = self.critic_fusion(fused_v).squeeze(-1)  # (T,)

            return log_probs, values, entropies
        else:
            # ---- Fallback: per-sample processing ----
            log_probs = []
            values = []
            entropies = []

            for t in range(T):
                logits = self._actor_logits(group_features_list[t])
                dist = Categorical(logits=logits)

                log_probs.append(dist.log_prob(actions[t]))
                entropies.append(dist.entropy())
                values.append(self._critic_value(state_features_batch[t]))

            return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


# =============================================================================
# PPO Trainer
# =============================================================================
class SwitcherPPO:
    """
    PPO trainer for the categorical group switcher.

    Follows the same structural pattern as robot_nav/models/PPO/PPO.py:
        policy + policy_old + optimizer + buffer + writer.

    Key differences from the continuous PPO:
        - Categorical distribution (no action_std / covariance).
        - Variable-size action space handled via per-sample processing.
        - Entropy bonus replaces action_std decay for exploration control.

    Args:
        embed_dim: Per-robot embedding dimension from GAT backbone output
            (d = 2 * GAT embedding_dim = 512).  ``h_g`` and ``h_glob``
            each have this dim.
        group_scalar_dim: Number of per-group scalar features.
        state_scalar_dim: Number of state-level scalar features for the value head.
        lr_actor: Learning rate for actor parameters.
        lr_critic: Learning rate for critic parameters.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        eps_clip: PPO clipping range.
        entropy_coeff: Entropy bonus coefficient (encourages exploration).
            Used as starting value when entropy annealing is enabled.
        entropy_coeff_end: Final entropy coefficient after annealing.
            If None, no annealing (constant entropy_coeff).
        entropy_anneal_updates: Number of updates over which to linearly
            anneal entropy_coeff from start to end value.  0 = no annealing.
        value_coeff: Value loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        device: Torch device string.
        save_every: Save checkpoint every N training updates.
        load_model: If True, load weights on init.
        save_directory: Directory for saving checkpoints.
        model_name: Base filename for checkpoints.
        load_directory: Directory to load checkpoints from.
        **net_kwargs: Additional keyword arguments passed to SwitcherActorCritic
            (embed_hidden, group_scalar_hidden, fusion_hidden, etc.).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        group_scalar_dim: int = 13,
        state_scalar_dim: int = 5,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        entropy_coeff: float = 0.01,
        entropy_coeff_end: Optional[float] = None,
        entropy_anneal_updates: int = 0,
        value_coeff: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        save_every: int = 10,
        load_model: bool = False,
        save_directory: Path = Path("robot_nav/models/MARL/switcher/checkpoint"),
        model_name: str = "SwitcherPPO",
        load_directory: Path = Path("robot_nav/models/MARL/switcher/checkpoint"),
        **net_kwargs,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_start = entropy_coeff
        self.entropy_coeff_end = (
            entropy_coeff_end if entropy_coeff_end is not None else entropy_coeff
        )
        self.entropy_anneal_updates = entropy_anneal_updates
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0

        self.buffer = SwitcherRolloutBuffer()

        # Current policy
        self.policy = SwitcherActorCritic(
            embed_dim=embed_dim,
            group_scalar_dim=group_scalar_dim,
            state_scalar_dim=state_scalar_dim,
            **net_kwargs,
        ).to(device)

        # Separate LRs for actor and critic (same pattern as PPO.py)
        actor_params = (
            list(self.policy.actor_embed_tower.parameters())
            + list(self.policy.actor_scalar_tower.parameters())
            + list(self.policy.actor_fusion.parameters())
        )
        critic_params = (
            list(self.policy.critic_embed_tower.parameters())
            + list(self.policy.critic_scalar_tower.parameters())
            + list(self.policy.critic_fusion.parameters())
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_params, "lr": lr_actor},
                {"params": critic_params, "lr": lr_critic},
            ]
        )

        # Old policy for computing importance-sampling ratio
        self.policy_old = SwitcherActorCritic(
            embed_dim=embed_dim,
            group_scalar_dim=group_scalar_dim,
            state_scalar_dim=state_scalar_dim,
            **net_kwargs,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.writer = SummaryWriter(comment=model_name)

    def get_action(
        self,
        group_features: torch.Tensor,
        state_features: torch.Tensor,
        explore: bool = True,
    ) -> int:
        """
        Select a group using the old policy and store the transition in the buffer.

        Call buffer.add(..., reward=..., is_terminal=...) separately after
        the environment returns the reward for this decision step.

        Args:
            group_features: (M, D) tensor on self.device.
            state_features: (S,) tensor on self.device.
            explore: If True, sample from distribution; else use argmax.

        Returns:
            Selected group index (int).
        """
        with torch.no_grad():
            action, log_prob, value = self.policy_old.act(
                group_features, state_features, sample=explore
            )

        # Keep tensors on GPU to avoid CPU↔GPU round-trips.
        # They will be used on-device during PPO training.
        self.buffer.group_features.append(group_features.detach())
        self.buffer.state_features.append(state_features.detach())
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_prob)
        self.buffer.values.append(value)

        return action

    def store_reward(self, reward: float, is_terminal: bool):
        """
        Store the reward and terminal flag for the most recent action.

        Must be called once after each get_action() call, after the
        environment has been stepped for selection_interval steps.

        Args:
            reward: Accumulated reward over the selection interval.
            is_terminal: Whether the episode ended during this interval.
        """
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)

    def train(self, replay_buffer=None, iterations: int = 10, batch_size: int = 0):
        """
        Run PPO update on the collected rollout buffer.

        Signature matches PPO.py for compatibility (replay_buffer and batch_size
        are unused; the full buffer is used as a single batch).

        Args:
            replay_buffer: Unused, kept for API compatibility.
            iterations: Number of PPO epochs per update.
            batch_size: Unused.
        """
        T = len(self.buffer.rewards)
        if T == 0:
            return

        # ----- Compute discounted returns using GAE -----
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        values = [v.item() for v in self.buffer.values]

        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1 or is_terminals[t]:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1.0 - is_terminals[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - is_terminals[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)

        # ----- Convert to tensors (directly on device) -----
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        actions_t = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)

        # Group/state features are already on device (kept on GPU in get_action)
        group_features_list = self.buffer.group_features
        state_features_batch = torch.stack(self.buffer.state_features)  # already on device

        # Normalize advantages
        if T > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ----- Entropy coefficient annealing -----
        if self.entropy_anneal_updates > 0:
            progress = min(self.iter_count / self.entropy_anneal_updates, 1.0)
            self.entropy_coeff = (
                self.entropy_coeff_start
                + (self.entropy_coeff_end - self.entropy_coeff_start) * progress
            )

        # ----- PPO update for K epochs -----
        av_loss = 0.0
        av_policy_loss = 0.0
        av_value_loss = 0.0
        av_entropy = 0.0

        for _ in range(iterations):
            # Re-evaluate under current policy
            log_probs, state_values, entropies = self.policy.evaluate(
                group_features_list, state_features_batch, actions_t
            )
            state_values = state_values.squeeze()

            # Importance-sampling ratio
            ratios = torch.exp(log_probs - old_logprobs.detach())

            # Clipped surrogate objective
            surr1 = ratios * advantages_t
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = F.mse_loss(state_values, returns_t)

            # Entropy bonus (higher entropy = more exploration)
            entropy_loss = -entropies.mean()

            # Total loss
            loss = (
                policy_loss
                + self.value_coeff * value_loss
                + self.entropy_coeff * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            av_loss += loss.item()
            av_policy_loss += policy_loss.item()
            av_value_loss += value_loss.item()
            av_entropy += entropies.mean().item()

        # ----- Post-update bookkeeping -----
        # Copy current policy to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

        # Logging
        self.iter_count += 1
        self.writer.add_scalar("train/total_loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/policy_loss", av_policy_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/value_loss", av_value_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/entropy", av_entropy / iterations, self.iter_count)
        self.writer.add_scalar("train/entropy_coeff", self.entropy_coeff, self.iter_count)
        self.writer.add_scalar("train/buffer_size", T, self.iter_count)

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename: str, directory: Path):
        """Save policy checkpoint."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter_count": self.iter_count,
            },
            directory / f"{filename}.pt",
        )

    def load(self, filename: str, directory: Path):
        """Load policy checkpoint."""
        checkpoint = torch.load(
            Path(directory) / f"{filename}.pt",
            map_location=lambda storage, loc: storage,
        )
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "iter_count" in checkpoint:
            self.iter_count = checkpoint["iter_count"]
        print(f"Loaded SwitcherPPO weights from: {directory}/{filename}.pt")

    def load_supervised_weights(self, checkpoint_path: str):
        """
        Load pre-trained supervised GroupSwitcher weights into the actor.

        Remaps supervised layer names to RL actor names:
            embed_tower  → actor_embed_tower
            scalar_tower → actor_scalar_tower
            fusion       → actor_fusion

        Critic weights remain randomly initialized.

        Args:
            checkpoint_path: Path to supervised training checkpoint (.pt).
                Expected format (from train_switcher.py):
                    {"model_state_dict": ..., "config": ..., ...}
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )

        # Handle both training checkpoint and raw state_dict
        if "model_state_dict" in checkpoint:
            sup_sd = checkpoint["model_state_dict"]
        elif "policy_state_dict" in checkpoint:
            sup_sd = checkpoint["policy_state_dict"]
        else:
            sup_sd = checkpoint

        # Remap supervised keys → actor keys
        name_map = {
            "embed_tower": "actor_embed_tower",
            "scalar_tower": "actor_scalar_tower",
            "fusion": "actor_fusion",
        }

        actor_sd = {}
        for key, value in sup_sd.items():
            prefix = key.split(".")[0]
            if prefix in name_map:
                new_key = key.replace(prefix, name_map[prefix], 1)
                actor_sd[new_key] = value

        if not actor_sd:
            print("WARNING: No matching weights found in supervised checkpoint!")
            return

        # Verify shape compatibility before loading
        current_sd = self.policy.state_dict()
        for key, value in actor_sd.items():
            if key in current_sd and current_sd[key].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': "
                    f"supervised={value.shape} vs RL={current_sd[key].shape}. "
                    f"Ensure scalar_dim matches (supervised: no urgency flag, "
                    f"scalar_dim=13)."
                )

        # Load into current policy (strict=False skips critic keys)
        missing, unexpected = self.policy.load_state_dict(actor_sd, strict=False)

        # Sync to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Report
        critic_missing = [k for k in missing if "critic" in k]
        other_missing = [k for k in missing if "critic" not in k]
        print(f"\nLoaded {len(actor_sd)} supervised weights into actor:")
        for k in sorted(actor_sd.keys()):
            print(f"  ✓ {k}  {actor_sd[k].shape}")
        if other_missing:
            print(f"  ⚠ Non-critic missing keys: {other_missing}")
        print(f"  ℹ Critic keys (randomly initialized): {len(critic_missing)}")

        if "config" in checkpoint:
            sup_cfg = checkpoint["config"]
            print(f"  ℹ Supervised config: embed_dim={sup_cfg.get('embed_dim')}, "
                  f"scalar_dim={sup_cfg.get('scalar_dim', 'N/A')}, "
                  f"epochs={sup_cfg.get('epochs')}")
        print()

    def freeze_actor(self):
        """Freeze actor parameters (for critic warm-up phase)."""
        for name, param in self.policy.named_parameters():
            if name.startswith("actor_"):
                param.requires_grad = False
        for name, param in self.policy_old.named_parameters():
            if name.startswith("actor_"):
                param.requires_grad = False
        print("Actor parameters frozen")

    def unfreeze_actor(self):
        """Unfreeze actor parameters (after critic warm-up)."""
        for name, param in self.policy.named_parameters():
            if name.startswith("actor_"):
                param.requires_grad = True
        for name, param in self.policy_old.named_parameters():
            if name.startswith("actor_"):
                param.requires_grad = True
        print("Actor parameters unfrozen")
