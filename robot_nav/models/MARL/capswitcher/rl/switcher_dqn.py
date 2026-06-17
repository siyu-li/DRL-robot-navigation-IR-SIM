"""
CAPSwitcher DQN implementation (off-policy).

Rationale
---------
The switcher is a *binary* decision (coarse vs. precise) over a *frozen*
backbone, observed as a fixed 512-dim mean-pooled embedding.  This is an ideal
setting for an off-policy value method: a replay buffer lets every collected
transition be reused for many gradient updates, so far fewer (expensive)
IR-SIM environment steps are needed to converge than with an on-policy method
like PPO that discards each rollout.  Because the cached embeddings are stored
in the buffer, the DQN updates are pure MLP passes — the GAT backbone runs only
during data collection, never during training.

Classes
-------
DeepSetsQNet
    Permutation-invariant Q-network mapping per-robot embeddings (N, 512) to
    Q-values over {coarse, precise}, via a Deep Sets readout (per-robot φ →
    sum⊕max aggregation → ρ).

ReplayBuffer
    Ring buffer of (obs, action, reward, next_obs, done) transitions, where
    each obs is a per-robot embedding stack (N, embed_dim).

SwitcherDQN
    Double-DQN trainer: epsilon-greedy action selection, target network,
    smooth-L1 TD loss, TensorBoard logging, checkpointing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.capswitcher.policies.deep_sets_head import DeepSetsHead


# ---------------------------------------------------------------------------
# DeepSetsQNet
# ---------------------------------------------------------------------------

class DeepSetsQNet(nn.Module):
    """
    Deep Sets Q-network for the binary CAPSwitcher.

    Reads the *unpooled* per-robot decoder-output embeddings and learns its own
    permutation-invariant aggregation, so it never relies on a lossy fixed pool
    of the navigation embeddings.

        Q(s, ·) = ρ( aggregate_i φ(h_i) )

    Args:
        embed_dim:   Per-robot input embedding dimension. Default 512.
        phi_dims:    Hidden widths of the per-robot encoder φ. Default (256, 128).
        rho_dims:    Hidden widths of the post-aggregation ρ. Default (128,).
        num_actions: Number of discrete actions. Default 2 (coarse/precise).
        aggregation: Deep Sets pooling. Default "sum_max".
    """

    def __init__(
        self,
        embed_dim: int = 512,
        phi_dims: Sequence[int] = (256, 128),
        rho_dims: Sequence[int] = (128,),
        num_actions: int = 2,
        aggregation: str = "sum_max",
    ) -> None:
        super().__init__()
        self.head = DeepSetsHead(
            embed_dim=embed_dim,
            phi_dims=phi_dims,
            rho_dims=rho_dims,
            out_dim=num_actions,
            aggregation=aggregation,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args: x (B, N, embed_dim). Returns: Q-values (B, num_actions)."""
        return self.head(x, mask)


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity ring buffer of transitions for off-policy training.

    Observations are per-robot embedding stacks of shape ``(num_robots,
    embed_dim)``, stored as float32; sampling returns tensors on the requested
    device.
    """

    def __init__(self, capacity: int, num_robots: int, embed_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, num_robots, embed_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_robots, embed_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Return a random minibatch as tensors on *device*."""
        idx = np.random.randint(0, self._size, size=batch_size)
        obs_t      = torch.as_tensor(self.obs[idx], device=device)
        act_t      = torch.as_tensor(self.actions[idx], device=device)
        rew_t      = torch.as_tensor(self.rewards[idx], device=device)
        next_obs_t = torch.as_tensor(self.next_obs[idx], device=device)
        done_t     = torch.as_tensor(self.dones[idx], device=device)
        return obs_t, act_t, rew_t, next_obs_t, done_t


# ---------------------------------------------------------------------------
# SwitcherDQN
# ---------------------------------------------------------------------------

class SwitcherDQN:
    """
    Double-DQN trainer for the CAPSwitcher binary switching policy.

    Only ``SwitcherQNet`` parameters are updated; the GAT backbone that
    produces the embeddings is fully frozen throughout training.

    Typical training loop (driven by the caller)::

        obs = env.reset()
        for step in range(total_steps):
            eps = dqn.epsilon(step)
            action = dqn.select_action(obs, eps)
            next_obs, reward, done, info = env.step(action)
            dqn.store(obs, action, reward, next_obs, done)
            obs = env.reset() if done else next_obs
            if step >= dqn.learn_starts and step % dqn.train_freq == 0:
                dqn.update()
            if step % dqn.target_update_freq == 0:
                dqn.update_target()

    Args:
        num_robots:         Number of robots (set N in the (N, embed_dim) obs).
        embed_dim:          Per-robot embedding dimension. Default 512.
        phi_dims:           Hidden widths of the Deep Sets per-robot encoder φ.
        rho_dims:           Hidden widths of the post-aggregation ρ.
        aggregation:        Deep Sets pooling. Default "sum_max".
        num_actions:        Discrete action count. Default 2.
        lr:                 Adam learning rate. Default 3e-4.
        gamma:              Discount factor. Default 0.99.
        buffer_size:        Replay buffer capacity. Default 50_000.
        batch_size:         Minibatch size for updates. Default 128.
        learn_starts:       Env steps to fill the buffer before updating.
        train_freq:         Gradient step every N env steps. Default 1.
        target_update_freq: Hard target sync every N env steps. Default 500.
        eps_start/eps_end:  Epsilon-greedy schedule endpoints.
        eps_decay_steps:    Steps to linearly decay eps_start → eps_end.
        max_grad_norm:      Gradient clipping norm. Default 10.0.
        double_dqn:         Use Double-DQN target. Default True.
        device:             Torch device.
        writer_comment:     TensorBoard run label suffix.
        save_dir:           Directory for checkpoint files.
        model_name:         Base filename for checkpoints.
    """

    def __init__(
        self,
        num_robots: int,
        embed_dim: int = 512,
        phi_dims: Sequence[int] = (256, 128),
        rho_dims: Sequence[int] = (128,),
        aggregation: str = "sum_max",
        num_actions: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        learn_starts: int = 1_000,
        train_freq: int = 1,
        target_update_freq: int = 500,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        max_grad_norm: float = 10.0,
        double_dqn: bool = True,
        device: torch.device = torch.device("cpu"),
        writer_comment: str = "CAPSwitcherDQN",
        save_dir: Path = Path(
            "robot_nav/models/MARL/capswitcher/checkpoints/cap_switcher"
        ),
        model_name: str = "capswitcher_dqn",
    ) -> None:
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_starts = learn_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.max_grad_norm = max_grad_norm
        self.double_dqn = double_dqn
        self.num_actions = num_actions
        self.device = device
        self.save_dir = Path(save_dir)
        self.model_name = model_name

        def _make_qnet() -> DeepSetsQNet:
            return DeepSetsQNet(
                embed_dim=embed_dim,
                phi_dims=phi_dims,
                rho_dims=rho_dims,
                num_actions=num_actions,
                aggregation=aggregation,
            ).to(device)

        self.q_net = _make_qnet()
        self.target_net = _make_qnet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, num_robots, embed_dim)

        self.writer = SummaryWriter(comment=f"_{writer_comment}")
        self.update_count: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def epsilon(self, step: int) -> float:
        """Linearly decayed epsilon for the given global env step."""
        frac = min(1.0, step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action over Q(obs, ·)."""
        if np.random.random() < epsilon:
            return int(np.random.randint(self.num_actions))
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.q_net(obs_t)
            return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> Optional[dict]:
        """
        Run one Double-DQN gradient step on a sampled minibatch.

        Returns:
            Loss/metric dict, or ``None`` if the buffer has too few samples.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs_t, act_t, rew_t, next_obs_t, done_t = self.buffer.sample(
            self.batch_size, self.device
        )

        # ---- TD target ------------------------------------------------
        with torch.no_grad():
            if self.double_dqn:
                # Online net selects the next action; target net evaluates it.
                next_actions = self.q_net(next_obs_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs_t).max(dim=1).values
            target = rew_t + self.gamma * next_q * (1.0 - done_t)

        # ---- Current Q(s, a) ------------------------------------------
        q = self.q_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        loss = nn.functional.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_count += 1
        return {
            "dqn/loss":      float(loss.item()),
            "dqn/q_mean":    float(q.mean().item()),
            "dqn/target_mean": float(target.mean().item()),
        }

    def update_target(self) -> None:
        """Hard-copy the online network weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def log(self, step: int, stats: dict) -> None:
        """Write a dict of scalar metrics to TensorBoard at *step*."""
        for k, v in stats.items():
            self.writer.add_scalar(k, v, step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, filename: Optional[str] = None) -> None:
        """Save ``SwitcherQNet`` weights to *save_dir*."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        name = filename or f"{self.model_name}_update{self.update_count}"
        path = self.save_dir / f"{name}.pth"
        torch.save(self.q_net.state_dict(), path)
        print(f"[SwitcherDQN] Saved: {path}")

    def load(self, path: str) -> None:
        """Load ``SwitcherQNet`` weights from *path* (also syncs target net)."""
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        print(f"[SwitcherDQN] Loaded: {path}")
