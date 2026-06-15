"""
Educational Exercise: TD3 Training Loop from Scratch
=====================================================

Goal: Replicate the essential structure of marl_train_obstacle_14robots.py
      WITHOUT a real environment. You will build:

  1.  AbstractEnv       — a fake environment that returns random tensors
  2.  Actor             — policy network  (nn.Module)
  3.  Critic            — twin Q-value network  (nn.Module)
  4.  ReplayBuffer      — experience storage  (collections.deque)
  5.  TD3Agent          — wraps actor/critic + training logic
  6.  main()            — the outer training loop + TensorBoard logging

Work through each TODO in order.  Run the script after every section to
verify it runs without errors before moving on.

Run command:
    python -m exercises.td3_training_exercise_blank

Expected output (after all TODOs are filled):
    Using device: cpu
    Starting training...
    Epoch 10 | Buffer: ... | Avg Reward: ...
    ...
    Training complete. Model saved.
"""

# =============================================================================
# 0. Imports
# =============================================================================
# PyTorch core: tensors, autograd, neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilities
import numpy as np
from collections import deque
import random
from pathlib import Path

# TensorBoard: log scalars so you can inspect with `tensorboard --logdir runs/`
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# 1. Abstract Environment
# =============================================================================
# In marl_train_obstacle_14robots.py, MARL_SIM_OBSTACLE plays this role.
# We abstract it here so you can focus on the training code.
#
# Contract: every RL environment must provide:
#   - reset()  -> state
#   - step(action) -> (next_state, reward, done)
#
# State shape:  (state_dim,)   — a flat vector of floats
# Action shape: (action_dim,)  — continuous action, clipped to [-1, 1]

class AbstractEnv:
    """
    Minimal fake environment for educational purposes.

    Internally it just returns random states and Gaussian rewards.
    You never need to understand its internals — only its interface.
    """

    def __init__(self, state_dim: int = 11, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._step_count = 0
        self._max_steps = 300

    def reset(self) -> np.ndarray:
        """Reset and return the initial state as a numpy array."""
        self._step_count = 0
        return np.random.randn(self.state_dim).astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Apply action and return (next_state, reward, done).

        Returns:
            next_state: np.ndarray of shape (state_dim,)
            reward:     float
            done:       bool — True if episode should end
        """
        self._step_count += 1
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        reward = float(-np.linalg.norm(action))  # reward: penalise large actions
        done = self._step_count >= self._max_steps
        return next_state, reward, done


# =============================================================================
# 2. Actor Network  (Policy)
# =============================================================================
# The actor maps a state to an action.
#
# Real equivalent: ActorObstacle in marlTD3_obstacle.py
#   — but here we skip the graph attention and use a plain MLP.
#
# Architecture:
#   Linear(state_dim → 256) → LeakyReLU
#   Linear(256 → 256)       → LeakyReLU
#   Linear(256 → action_dim) → Tanh     ← clips output to (-1, 1)

class Actor(nn.Module):
    """
    Simple MLP policy network.

    Args:
        state_dim:  Dimension of the observation vector.
        action_dim: Dimension of the continuous action vector.
    """

    def __init__(self, state_dim: int, action_dim: int):
        # CONCEPT: every nn.Module subclass MUST call super().__init__()
        #          before defining any layers.
        super().__init__()

        # TODO-1: Define three nn.Linear layers as described in the docstring above.
        #         Store them as self.l1, self.l2, self.l3.
        #         Hint: nn.Linear(in_features, out_features)
        # -----------------------------------------------------------------------
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        # -----------------------------------------------------------------------

        # TODO-2: Apply Kaiming (He) initialisation to each layer's weights.
        #         Use nonlinearity="leaky_relu" since we use LeakyReLU activations.
        #         Also initialise biases to zero.
        #
        #         Pattern from the real codebase:
        #             nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        #             nn.init.zeros_(layer.bias)
        # -----------------------------------------------------------------------
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias)
        # -----------------------------------------------------------------------

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state  →  action.

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            action: Tensor of shape (batch_size, action_dim), values in (-1, 1).
        """
        # TODO-3: Implement the forward pass using the layers you defined.
        #         Use F.leaky_relu() for the first two layers.
        #         Use torch.tanh() for the final output.
        # -----------------------------------------------------------------------
        x1 = F.leaky_relu(self.l1(state))
        x2 = F.leaky_relu(self.l2(x1))
        action = torch.tanh(self.l3(x2))
        return action
        # -----------------------------------------------------------------------


# =============================================================================
# 3. Critic Network  (Twin Q-values)
# =============================================================================
# The critic maps (state, action) → Q-value (expected future reward).
#
# TD3 uses TWO independent critics (Q1 and Q2) to reduce overestimation bias.
# During target computation we take min(Q1, Q2).
#
# Architecture (same for both Q-networks):
#   — state  branch: Linear(state_dim → 256) → LeakyReLU
#   — action branch: Linear(action_dim → 256)
#   — merge by addition (same as real codebase: layer_2_s(s) + layer_2_a(a))
#   — Linear(256 → 256) → LeakyReLU
#   — Linear(256 → 1)

class Critic(nn.Module):
    """
    Twin Q-value critic.

    Call forward(state, action) to get (Q1, Q2).
    Call Q1(state, action) during actor update to avoid unnecessary computation.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # TODO-4: Define layers for both Q1 and Q2 networks.
        #         Q1 layer names: q1_l1, q1_l2_s, q1_l2_a, q1_l3, q1_l4
        #         Q2 layer names: q2_l1, q2_l2_s, q2_l2_a, q2_l3, q2_l4
        #
        #         q*_l1:   Linear(state_dim → 256)
        #         q*_l2_s: Linear(256 → 256)         ← state stream after first layer
        #         q*_l2_a: Linear(action_dim → 256)   ← action stream merged here
        #         q*_l3:   Linear(512 → 256)
        #         q*_l4:   Linear(256 → 1)            ← scalar Q-value
        #
        #         Then apply Kaiming init + zero bias to all layers using:
        #             for layer in self.modules():
        #                 if isinstance(layer, nn.Linear): ...
        # -----------------------------------------------------------------------
        self.q1_l1 = nn.Linear(state_dim, 256)
        self.q1_l2_s = nn.Linear(256, 256)
        self.q1_l2_a = nn.Linear(action_dim, 256)
        self.q1_l3 = nn.Linear(512, 256)
        self.q1_l4 = nn.Linear(256, 1)
        
        self.q2_l1 = nn.Linear(state_dim, 256)
        self.q2_l2_s = nn.Linear(256, 256)
        self.q2_l2_a = nn.Linear(action_dim, 256)
        self.q2_l3 = nn.Linear(512, 256)
        self.q2_l4 = nn.Linear(256, 1)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(layer.bias)
        # -----------------------------------------------------------------------

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both Q-values.

        Args:
            state:  (batch_size, state_dim)
            action: (batch_size, action_dim)

        Returns:
            (Q1, Q2): each of shape (batch_size, 1)
        """
        # TODO-5: Implement the forward pass for Q1, then repeat for Q2.
        #
        #         Q1 pass:
        #           s1 = leaky_relu(q1_l1(state))
        #           s1 = leaky_relu(q1_l2_s(s1) + q1_l2_a(action))  ← merge streams
        #           s1 = leaky_relu(q1_l3(s1))
        #           q1 = q1_l4(s1)
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-5")
        # -----------------------------------------------------------------------

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute only Q1 — used during actor update to save compute."""
        # Hint: same as the Q1 block in forward(), just return q1 directly.
        raise NotImplementedError("Q1 helper")


# =============================================================================
# 4. Replay Buffer
# =============================================================================
# Off-policy algorithms (TD3) are data-efficient because they reuse past
# experience stored in the replay buffer.
#
# Each entry is a transition: (state, action, reward, done, next_state)
# Training randomly samples a mini-batch from this buffer.
#
# Real equivalent: ReplayBufferObstacle in replay_buffer_obstacle.py

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer using collections.deque.

    When full, the oldest transition is automatically discarded (maxlen).
    """

    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.buffer: deque = deque(maxlen=capacity)
        random.seed(seed)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_state: np.ndarray,
    ) -> None:
        """Store one transition."""
        # CONCEPT: we store numpy arrays here (CPU, small memory footprint).
        #          They are converted to tensors only when sampled for training.
        self.buffer.append((state, action, reward, done, next_state))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch of transitions.

        TODO-6: Implement this method.
                1. Use random.sample(self.buffer, batch_size) to draw transitions.
                2. Unzip the list of tuples into five separate lists.
                3. Convert each list to a numpy array with np.array(..., dtype=np.float32).
                4. Return (states, actions, rewards, dones, next_states).
                   rewards and dones should be reshaped to (-1, 1).

                Hint — unzipping a list of tuples:
                    pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
                    xs, ys = zip(*pairs)   # xs=(1,2,3), ys=('a','b','c')

        Returns:
            states:      np.ndarray  (batch_size, state_dim)
            actions:     np.ndarray  (batch_size, action_dim)
            rewards:     np.ndarray  (batch_size, 1)
            dones:       np.ndarray  (batch_size, 1)
            next_states: np.ndarray  (batch_size, state_dim)
        """
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-6")
        # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# 5. TD3 Agent
# =============================================================================
# The agent owns:
#   - actor  + actor_target   (policy + its slowly-updated copy)
#   - critic + critic_target  (Q-function + its slowly-updated copy)
#   - optimizers
#   - TensorBoard writer
#   - save / load methods
#
# Real equivalent: TD3Obstacle class in marlTD3_obstacle.py

class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

    Key hyperparameters:
        discount    (γ)   — how much future rewards are worth today
        tau         (τ)   — soft update rate for target networks  (0 < τ << 1)
        policy_freq       — actor is updated only every N critic updates
        policy_noise      — Gaussian noise on target actions (smoothing)
        noise_clip        — clip range for target action noise
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        device: str = "cpu",
        save_dir: str = "exercises/checkpoints",
        model_name: str = "td3_exercise",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(device)
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ---- Networks --------------------------------------------------------
        # CONCEPT: target networks start as exact copies of the online networks.
        #          They are updated slowly (soft update) for training stability.
        self.actor        = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # copy weights

        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ---- Optimizers ------------------------------------------------------
        # CONCEPT: Adam is the standard optimizer choice.
        #          Actor and critic use different learning rates.
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # ---- TensorBoard -----------------------------------------------------
        # CONCEPT: SummaryWriter opens a log directory.
        #          Every call to add_scalar(tag, value, step) appends a data point.
        #          View with: tensorboard --logdir runs/
        self.writer = SummaryWriter(comment=f"_{model_name}")
        self.iter_count = 0

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Choose an action for the given state.

        Steps:
          1. Convert numpy state → torch FloatTensor, add batch dim, move to self.device
          2. Run self.actor inside torch.no_grad()   ← disables gradient tracking
          3. Convert output tensor back to numpy: .cpu().numpy().flatten()
          4. Optionally add Gaussian exploration noise, then clip to [-max_action, max_action]

        TODO-7: Implement this method.
                Pattern from the real codebase (act() in TD3Obstacle):
                    state_t = torch.Tensor(state).to(self.device)
                    with torch.no_grad():
                        action = self.actor(state_t)
                    action_np = action.cpu().numpy().flatten()
        """
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-7")
        # -----------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train_step(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 64,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
    ) -> dict:
        """
        One TD3 gradient step.

        This is the most important method.  Read each phase carefully.

        Returns:
            dict with "critic_loss" and optionally "actor_loss"
        """
        # ---- Phase A: Sample from replay buffer & convert to tensors --------
        #
        # CONCEPT: numpy arrays (CPU float32) are converted to torch Tensors
        #          and moved to the compute device (GPU or CPU).

        states, actions, rewards, dones, next_states = replay_buffer.sample(batch_size)

        # TODO-8: Convert each numpy array to a torch.FloatTensor on self.device.
        #         Pattern: torch.FloatTensor(arr).to(self.device)
        #         Name them: s, a, r, d, s_
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-8")
        # -----------------------------------------------------------------------

        # ---- Phase B: Compute TD target (Bellman equation) ------------------
        #
        # CONCEPT: torch.no_grad() tells PyTorch not to build a computation graph
        #          for these operations.  The target is treated as a fixed number
        #          ("detached from the graph"), not something to differentiate.
        #
        #          Bellman equation:
        #            target = r + γ * (1-done) * min(Q1', Q2')(s', noisy_a')
        #
        #          The noisy next-action prevents Q overfitting (TD3 trick).

        with torch.no_grad():
            # TODO-9: Compute the noisy target action.
            #         1. next_a = actor_target(s_)
            #         2. noise  = normal tensor, scaled by policy_noise, clipped to
            #                     [-noise_clip, noise_clip]
            #                     Hint: torch.randn_like(next_a) gives same-shape noise
            #         3. next_a = clamp(next_a + noise, -max_action, max_action)
            # -------------------------------------------------------------------
            raise NotImplementedError("TODO-9")
            # -------------------------------------------------------------------

            # TODO-10: Compute the Bellman target Q value.
            #          1. (Q1_next, Q2_next) = critic_target(s_, next_a)
            #          2. target_Q = r + (1 - d) * discount * torch.min(Q1_next, Q2_next)
            # -------------------------------------------------------------------
            raise NotImplementedError("TODO-10")
            # -------------------------------------------------------------------

        # ---- Phase C: Update Critic -----------------------------------------
        #
        # CONCEPT: The standard PyTorch update pattern is always:
        #   1. optimizer.zero_grad()  — clear accumulated gradients from last step
        #   2. loss.backward()        — compute new gradients via backprop
        #   3. clip_grad_norm_()      — (optional safety) prevents exploding gradients
        #   4. optimizer.step()       — apply gradients to update weights

        current_Q1, current_Q2 = self.critic(s, a)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # TODO-11: Apply the update pattern to self.critic_optimizer.
        #          Use torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 7.0)
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-11")
        # -----------------------------------------------------------------------

        metrics = {"critic_loss": critic_loss.item()}

        # ---- Phase D: Update Actor (delayed) --------------------------------
        #
        # CONCEPT: "Delayed" means we update the actor less often than the critic
        #          (every policy_freq=2 iterations).  This prevents the actor from
        #          chasing a noisy Q estimate.
        #
        # The actor loss is the NEGATIVE of Q1 (we want to maximise Q):
        #   actor_loss = -Q1(s, actor(s)).mean()
        #
        # IMPORTANT: We use self.critic.Q1() here, NOT the target critic.
        #            The actor gradient must flow through the *online* critic.

        actor_loss = None
        if self.iter_count % policy_freq == 0:

            # TODO-12: Compute actor_loss = negative mean of Q1(s, actor(s)).
            # -------------------------------------------------------------------
            raise NotImplementedError("TODO-12")
            # -------------------------------------------------------------------

            # TODO-13: Apply the update pattern to self.actor_optimizer.
            # -------------------------------------------------------------------
            raise NotImplementedError("TODO-13")
            # -------------------------------------------------------------------

            # ---- Phase E: Soft update of target networks --------------------
            #
            # CONCEPT: Polyak averaging keeps target networks stable.
            #   θ_target = τ * θ_online + (1-τ) * θ_target
            #
            # τ=0.005 means the target changes only 0.5% per update.
            # This is much more stable than hard-copying weights every N steps.

            # TODO-14: Implement soft update for BOTH actor and critic targets.
            #          Iterate over zip(network.parameters(), target.parameters())
            #          and use: target_param.data.copy_(τ * param + (1-τ) * target_param)
            # -------------------------------------------------------------------
            raise NotImplementedError("TODO-14")
            # -------------------------------------------------------------------

            metrics["actor_loss"] = actor_loss.item()

        # ---- Phase F: Log to TensorBoard ------------------------------------
        #
        # CONCEPT: add_scalar(tag, scalar_value, global_step)
        #          tag uses "/" to organise into groups in the TensorBoard UI.
        #          global_step is used as the x-axis.

        # TODO-15: Log critic_loss and actor_loss (if computed) via self.writer.
        #          Use tags "train/critic_loss" and "train/actor_loss".
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-15")
        # -----------------------------------------------------------------------

        self.iter_count += 1
        return metrics

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------
    #
    # CONCEPT: state_dict() returns an OrderedDict of {layer_name: weight_tensor}.
    #          We save the state_dicts of all four networks plus iter_count.
    #          On load, map_location=self.device ensures a GPU checkpoint can be
    #          loaded on a CPU machine.

    def save(self, name: str = "checkpoint") -> None:
        """
        Save actor, actor_target, critic, critic_target weights to disk.

        TODO-16: Use torch.save() to save each network's state_dict.
                 Save to:  self.save_dir / f"{name}_{network_name}.pth"
                 Also save {"iter_count": self.iter_count} to a meta file.

                 Pattern:
                     torch.save(self.actor.state_dict(), self.save_dir / f"{name}_actor.pth")
        """
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-16")
        # -----------------------------------------------------------------------

    def load(self, name: str = "checkpoint") -> None:
        """
        Load previously saved weights.

        TODO-17: Mirror save() using torch.load() + load_state_dict().
                 Always pass map_location=self.device to torch.load().
        """
        # -----------------------------------------------------------------------
        raise NotImplementedError("TODO-17")
        # -----------------------------------------------------------------------


# =============================================================================
# 6. Main Training Loop
# =============================================================================
# This mirrors the while-loop in marl_train_obstacle_14robots.py:
#
#   while epoch < max_epochs:
#       action = agent.select_action(state)
#       next_state, reward, done = env.step(action)
#       buffer.add(state, action, reward, done, next_state)
#       if done or max_steps reached:
#           env.reset()
#           if buffer is large enough:
#               for _ in range(TRAIN_ITERS):
#                   agent.train_step(buffer)
#               log metrics

def main():
    # ---- Hyperparameters ----
    STATE_DIM        = 11
    ACTION_DIM       = 2
    MAX_EPOCHS       = 500          # number of completed episodes
    MAX_STEPS        = 300          # max steps per episode before forced reset
    BATCH_SIZE       = 64
    BUFFER_SIZE      = 100_000
    TRAIN_EVERY_N    = 10           # only start training after N episodes of data
    TRAIN_ITERS      = 10           # gradient steps per training call
    CHECKPOINT_EVERY = 100          # save checkpoint every N epochs
    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")

    # ---- Instantiate components ----
    env    = AbstractEnv(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    buffer = ReplayBuffer(capacity=BUFFER_SIZE)
    agent  = TD3Agent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=DEVICE,
        model_name="td3_exercise",
    )

    # ---- Initial state ----
    state = env.reset()

    # ---- Accumulators for episode-level metrics ----
    running_reward    = 0.0
    running_timesteps = 0
    episode_count     = 0
    epoch             = 0
    steps             = 0

    print("Starting training...")

    # =========================================================================
    # TODO-18: Understand the training loop below by reading every comment.
    #          Trace through ONE complete iteration on paper:
    #            a) agent.select_action  →  action
    #            b) env.step(action)     →  next_state, reward, done
    #            c) buffer.add(...)      →  stored transition
    #            d) episode end          →  agent.train_step called
    #
    #          Then fill in TODO-19 inside the loop.
    # =========================================================================

    while epoch < MAX_EPOCHS:
        # 1. Select action (with exploration noise during training)
        action = agent.select_action(state, add_noise=True)

        # 2. Step the environment
        next_state, reward, done = env.step(action)

        # 3. Store transition in replay buffer
        # CONCEPT: we store numpy arrays — cheap, no device needed
        buffer.add(state, action, reward, float(done), next_state)

        # 4. Accumulate metrics
        running_reward    += reward
        running_timesteps += 1
        steps             += 1
        episode_count     += 1

        # 5. Move to next state
        state = next_state

        # 6. Episode termination: done flag OR max steps per episode
        if done or steps >= MAX_STEPS:
            state = env.reset()
            steps = 0
            epoch += 1

            # 7. Train when enough data has been collected
            if episode_count >= TRAIN_EVERY_N and len(buffer) >= BATCH_SIZE:

                avg_reward = running_reward / max(running_timesteps, 1)

                # TODO-19: Log "run/avg_reward" and "run/buffer_size" to agent.writer.
                #          Use agent.writer.add_scalar(tag, value, step).
                #          Use agent.iter_count as the global step.
                # -------------------------------------------------------------------
                raise NotImplementedError("TODO-19")
                # -------------------------------------------------------------------

                # Reset accumulators
                running_reward    = 0.0
                running_timesteps = 0
                episode_count     = 0

                # Multiple gradient steps per training call
                for _ in range(TRAIN_ITERS):
                    agent.train_step(buffer, batch_size=BATCH_SIZE)

                # Checkpoint save
                if epoch % CHECKPOINT_EVERY == 0:
                    agent.save(name=f"{agent.model_name}_epoch{epoch}")

                # Console log every 10 epochs
                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:4d}/{MAX_EPOCHS} | "
                        f"Buffer: {len(buffer):6d} | "
                        f"Avg Reward: {avg_reward:+.3f} | "
                        f"Train steps: {agent.iter_count}"
                    )

    # ---- Final save ----
    agent.save(name=agent.model_name)
    agent.writer.close()
    print("\nTraining complete. Model saved.")
    print(f"View logs: tensorboard --logdir runs/")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()


# =============================================================================
# BONUS CHALLENGES  (after completing all TODOs above)
# =============================================================================
#
# B1. LEARNING RATE SCHEDULER
#     Add a CosineAnnealingLR scheduler to the actor optimizer.
#     Call scheduler.step() at the end of each epoch.
#     Log the current LR with: writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
#
# B2. VALIDATION / EVALUATION
#     Add an evaluate() method to TD3Agent that runs the policy WITHOUT noise
#     for N episodes and returns the mean total reward.
#     Call it every 50 epochs and log "eval/mean_reward".
#
# B3. RESUME FROM CHECKPOINT
#     Modify main() to accept a --resume flag that calls agent.load() before
#     the training loop.
#
# B4. MULTIPLE TRAINING ITERATIONS
#     In the real marl_train script, model.train() runs 80 gradient steps.
#     Profile the wall-clock time of your train_step with:
#         import time; t0 = time.time(); ...; print(time.time()-t0)
#     Then try batching differently and compare.
#
# B5. DTYPE AWARENESS
#     Change torch.FloatTensor → torch.as_tensor(..., dtype=torch.float32)
#     and verify the behaviour is the same.  When would you use float16?
