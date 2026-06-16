"""
CAPSwitcher PPO implementation.

Classes
-------
SwitcherActorCritic
    Shared-trunk actor + critic network.  Input is the (512,) mean-pooled
    pre-decoder embedding; outputs are logits over {coarse, precise} and
    a scalar state value.

SwitcherRolloutBuffer
    Fixed-length buffer that stores one complete rollout and converts it to
    tensors for PPO training.

SwitcherPPO
    Orchestrates rollout collection, GAE advantage estimation, and PPO
    gradient updates.  Logs all metrics to TensorBoard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# SwitcherActorCritic
# ---------------------------------------------------------------------------

class SwitcherActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for the binary CAPSwitcher.

    Both the actor (policy) head and the critic (value) head read from a
    shared two-layer trunk that processes the (B, embed_dim) swarm embedding.

    Architecture:
        trunk:       Linear(embed_dim → hidden1) → ReLU
                     Linear(hidden1   → hidden2) → ReLU
        actor_head:  Linear(hidden2 → num_actions)   → logits
        critic_head: Linear(hidden2 → 1)              → V(s)

    Args:
        embed_dim:   Input embedding dimension. Default 512.
        hidden1:     First hidden layer width.  Default 256.
        hidden2:     Second hidden layer width. Default 128.
        num_actions: Number of discrete actions. Default 2 (coarse/precise).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden1: int = 256,
        hidden2: int = 128,
        num_actions: int = 2,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden2, num_actions)
        self.critic_head = nn.Linear(hidden2, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, embed_dim) swarm embedding.

        Returns:
            logits: (B, num_actions) raw action logits.
            value:  (B,) estimated state values.
        """
        feat = self.trunk(x)
        return self.actor_head(feat), self.critic_head(feat).squeeze(-1)

    def get_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy.

        Args:
            obs: (B, embed_dim) or (embed_dim,) observation.

        Returns:
            action:   (B,) sampled discrete actions.
            log_prob: (B,) log-probabilities of the sampled actions.
            value:    (B,) critic estimates.
            entropy:  (B,) policy distribution entropy.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate stored (obs, action) pairs under the current policy.

        Returns:
            log_prob: (B,) log-probabilities.
            value:    (B,) critic estimates.
            entropy:  (B,) distribution entropy.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()


# ---------------------------------------------------------------------------
# SwitcherRolloutBuffer
# ---------------------------------------------------------------------------

class SwitcherRolloutBuffer:
    """
    Fixed-length rollout buffer for one PPO update cycle.

    Stores raw Python scalars/arrays and converts them to GPU tensors on
    demand via ``to_tensors()``.
    """

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)

    def to_tensors(
        self, device: torch.device
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return all stored data as tensors on *device*."""
        obs_t  = torch.as_tensor(
            np.array(self.obs), dtype=torch.float32, device=device
        )
        act_t  = torch.as_tensor(
            np.array(self.actions), dtype=torch.long, device=device
        )
        lp_t   = torch.as_tensor(
            np.array(self.log_probs), dtype=torch.float32, device=device
        )
        rew_t  = torch.as_tensor(
            np.array(self.rewards), dtype=torch.float32, device=device
        )
        val_t  = torch.as_tensor(
            np.array(self.values), dtype=torch.float32, device=device
        )
        done_t = torch.as_tensor(
            np.array(self.dones, dtype=np.float32), dtype=torch.float32, device=device
        )
        return obs_t, act_t, lp_t, rew_t, val_t, done_t


# ---------------------------------------------------------------------------
# SwitcherPPO
# ---------------------------------------------------------------------------

class SwitcherPPO:
    """
    PPO trainer for the CAPSwitcher binary switching policy.

    Only ``SwitcherActorCritic`` parameters are updated; the GAT backbone
    that produces the embeddings is fully frozen throughout training.

    Training flow per update:
        1. ``collect_rollout(env, rollout_steps)`` — run current policy for
           *rollout_steps* switcher decisions, store transitions.
        2. ``train(buffer, last_obs)`` — compute GAE advantages, run PPO
           mini-batch epochs, return loss statistics.
        3. ``log(rollout_stats, loss_stats)`` — write scalars to TensorBoard.
        4. ``save()`` — checkpoint the SwitcherActorCritic weights.

    Args:
        embed_dim:       Input embedding dimension. Default 512.
        hidden1:         First hidden layer width. Default 256.
        hidden2:         Second hidden layer width. Default 128.
        lr:              Adam learning rate. Default 3e-4.
        gamma:           Discount factor. Default 0.99.
        lam:             GAE-λ. Default 0.95.
        clip_eps:        PPO clipping ε. Default 0.2.
        ppo_epochs:      PPO update epochs per rollout. Default 10.
        batch_size:      Mini-batch size. Default 64.
        ent_coef:        Entropy bonus coefficient. Default 0.01.
        vf_coef:         Value loss coefficient. Default 0.5.
        max_grad_norm:   Gradient clipping norm. Default 0.5.
        device:          Torch device.
        writer_comment:  TensorBoard run label suffix.
        save_dir:        Directory for checkpoint files.
        model_name:      Base filename for checkpoints.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden1: int = 256,
        hidden2: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
        writer_comment: str = "CAPSwitcher",
        save_dir: Path = Path(
            "robot_nav/models/MARL/capswitcher/checkpoints/cap_switcher"
        ),
        model_name: str = "capswitcher",
    ) -> None:
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.save_dir = Path(save_dir)
        self.model_name = model_name

        self.policy = SwitcherActorCritic(
            embed_dim=embed_dim,
            hidden1=hidden1,
            hidden2=hidden2,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.writer = SummaryWriter(comment=f"_{writer_comment}")
        self.update_count: int = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self, env, rollout_steps: int
    ) -> tuple[SwitcherRolloutBuffer, dict]:
        """
        Collect *rollout_steps* switcher decisions using the current policy.

        Resets the environment automatically at episode boundaries.

        Args:
            env:           SwitcherEnv instance.
            rollout_steps: Number of switcher decisions to collect.

        Returns:
            buffer:        Filled ``SwitcherRolloutBuffer``.
            stats:         Episode-level statistics dict for logging.
        """
        buffer = SwitcherRolloutBuffer()
        self.policy.eval()

        obs = env.reset()
        ep_reward: float = 0.0
        ep_len: int = 0
        coarse_count: int = 0
        num_episodes: int = 0
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        collision_count: int = 0
        success_count: int = 0

        for _ in range(rollout_steps):
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value, _ = self.policy.get_action(obs_t)

            action_int = int(action.item())
            log_prob_f = float(log_prob.item())
            value_f = float(value.item())

            if action_int == 0:
                coarse_count += 1

            next_obs, reward, done, info = env.step(action_int)

            buffer.add(obs, action_int, log_prob_f, reward, value_f, done)
            ep_reward += reward
            ep_len += 1
            obs = next_obs

            if done:
                if info.get("collision"):
                    collision_count += 1
                if info.get("all_reached"):
                    success_count += 1
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_len)
                ep_reward = 0.0
                ep_len = 0
                num_episodes += 1
                obs = env.reset()

        stats = {
            "mean_reward": (
                float(np.mean(episode_rewards)) if episode_rewards else ep_reward
            ),
            "mean_ep_length": (
                float(np.mean(episode_lengths)) if episode_lengths else float(ep_len)
            ),
            "frac_coarse": coarse_count / max(len(buffer), 1),
            "success_rate": success_count / max(num_episodes, 1),
            "collision_rate": collision_count / max(num_episodes, 1),
            "num_episodes": num_episodes,
        }
        return buffer, stats

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def train(
        self,
        buffer: SwitcherRolloutBuffer,
        last_obs: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run a full PPO update on the collected rollout.

        Args:
            buffer:   Filled ``SwitcherRolloutBuffer``.
            last_obs: Final observation after the rollout, used to bootstrap
                      the GAE return.  Pass ``None`` if the episode ended
                      (bootstrap value = 0).

        Returns:
            loss_stats: Dict with mean loss values for TensorBoard.
        """
        obs_t, act_t, old_lp_t, rew_t, val_t, done_t = buffer.to_tensors(
            self.device
        )

        # ---- Compute GAE advantages ------------------------------------
        with torch.no_grad():
            if last_obs is not None:
                last_obs_t = torch.as_tensor(
                    last_obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                _, last_value = self.policy(last_obs_t)
                last_value = last_value.squeeze()
            else:
                last_value = torch.tensor(0.0, device=self.device)

        advantages = self._compute_gae(rew_t, val_t, done_t, last_value)
        returns = advantages + val_t

        # Normalize advantages for stable training.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO mini-batch epochs ------------------------------------
        self.policy.train()
        T = len(buffer)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        total_losses: list[float] = []

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) < 2:
                    continue  # skip tiny trailing batch

                b_obs  = obs_t[idx]
                b_act  = act_t[idx]
                b_old_lp = old_lp_t[idx]
                b_adv  = advantages[idx]
                b_ret  = returns[idx]

                new_lp, new_val, entropy = self.policy.evaluate(b_obs, b_act)

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * b_adv
                )
                policy_loss  = -torch.min(surr1, surr2).mean()
                value_loss   = nn.functional.mse_loss(new_val, b_ret)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())

        return {
            "ppo/policy_loss": float(np.mean(policy_losses)),
            "ppo/value_loss":  float(np.mean(value_losses)),
            # Report positive entropy (negate the negative entropy loss)
            "ppo/entropy":     float(-np.mean(entropy_losses)),
            "ppo/total_loss":  float(np.mean(total_losses)),
        }

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def log(self, rollout_stats: dict, loss_stats: dict) -> None:
        """Write all statistics to TensorBoard at the current update step."""
        step = self.update_count
        self.writer.add_scalar("rollout/mean_reward",    rollout_stats["mean_reward"],    step)
        self.writer.add_scalar("rollout/mean_ep_length", rollout_stats["mean_ep_length"], step)
        self.writer.add_scalar("switcher/frac_coarse",   rollout_stats["frac_coarse"],    step)
        self.writer.add_scalar("episode/success_rate",   rollout_stats["success_rate"],   step)
        self.writer.add_scalar("episode/collision_rate", rollout_stats["collision_rate"], step)
        for k, v in loss_stats.items():
            self.writer.add_scalar(k, v, step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, filename: Optional[str] = None) -> None:
        """Save ``SwitcherActorCritic`` weights to *save_dir*."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        name = filename or f"{self.model_name}_update{self.update_count}"
        path = self.save_dir / f"{name}.pth"
        torch.save(self.policy.state_dict(), path)
        print(f"[SwitcherPPO] Saved: {path}")

    def load(self, path: str) -> None:
        """Load ``SwitcherActorCritic`` weights from *path*."""
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        print(f"[SwitcherPPO] Loaded: {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE-λ)."""
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        return advantages
