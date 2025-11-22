"""
PPO (Proximal Policy Optimization) Implementation for Atari Pong
===============================================================

This module implements PPO, a state-of-the-art deep reinforcement learning algorithm.
PPO is an on-policy method that uses a clipped objective to prevent destructive policy updates.

Key components:
1. Neural Networks: CNN feature extractor + actor-critic heads
2. Experience Collection: Gather rollouts from environment
3. Advantage Estimation: Use GAE (Generalized Advantage Estimation)
4. Policy Update: Apply clipped PPO loss
"""

import ale_py  # Needed for Atari envs under gymnasium[atari]
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import json
import os
from typing import Tuple, Dict, List

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting features from Atari game frames.

    Input: Single grayscale frame (210x160) → shape (batch, 1, 210, 160)
    Output: Feature vector of size 512 → shape (batch, 512)

    Uses raw 210x160 grayscale images without resizing, as specified by professor.
    """

    def __init__(self, input_channels: int = 1, feature_dim: int = 512):
        super(CNNFeatureExtractor, self).__init__()

        # First convolutional layer: 1 channel (grayscale) → 16 filters
        # kernel_size=5, stride=2 → reduces spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=0,
        )

        # Second convolutional layer: 16 → 32 filters
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0
        )

        # Third convolutional layer: 32 → 64 filters
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
        )

        # Manual conv output size computation for 210x160 input:
        # Conv1 (k=5, s=2): 210→103, 160→78
        # Conv2 (k=5, s=2): 103→50, 78→37
        # Conv3 (k=3, s=1): 50→48, 37→35
        conv_output_size = 64 * 48 * 35

        # Fully connected layer: flatten conv output → 512 features
        self.fc = nn.Linear(in_features=conv_output_size, out_features=feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN feature extractor.

        Args:
            x: (batch_size, 1, 210, 160)

        Returns:
            (batch_size, 512)
        """
        # Safety check to catch shape bugs early
        # Learn more: "NCHW format in PyTorch"
        assert x.dim() == 4, f"Expected 4D input (N, C, H, W), got {x.shape}"
        assert x.size(1) == 1, f"Expected 1 channel (grayscale), got {x.size(1)}"

        x = F.relu(self.conv1(x))  # (N,16,*,*)
        x = F.relu(self.conv2(x))  # (N,32,*,*)
        x = F.relu(self.conv3(x))  # (N,64,*,*)

        # Flatten: (N, 64, 48, 35) → (N, 64*48*35)
        x = x.view(x.size(0), -1)

        # Fully connected layer + ReLU
        x = F.relu(self.fc(x))  # (N,512)

        return x


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared CNN → policy head (actor) + value head (critic).
    """

    def __init__(self, num_actions: int = 3, feature_dim: int = 512):
        """
        Args:
            num_actions: Number of discrete actions
            feature_dim: Dimension of shared CNN feature vector
        """
        super(PPOActorCritic, self).__init__()

        # Shared CNN feature extractor for grayscale frames
        self.feature_extractor = CNNFeatureExtractor(
            input_channels=1, feature_dim=feature_dim
        )

        # Policy head: 512 → num_actions (logits)
        self.policy_head = nn.Linear(in_features=feature_dim, out_features=num_actions)

        # Value head: 512 → 1 (state value)
        self.value_head = nn.Linear(in_features=feature_dim, out_features=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch_size, 1, 210, 160)

        Returns:
            action_logits: (batch_size, num_actions)
            value: (batch_size, 1)
        """
        features = self.feature_extractor(x)  # (N,512)
        action_logits = self.policy_head(features)  # (N,num_actions)
        value = self.value_head(features)  # (N,1)
        return action_logits, value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy or evaluate a given action.

        Args:
            x: (batch_size, 1, 210, 160)
            action: optional tensor of actions

        Returns:
            action, log_prob, entropy, value
        """
        action_logits, value = self.forward(x)  # forward through net

        # Convert logits → probabilities
        action_probs = F.softmax(action_logits, dim=-1)  # Learn more: "softmax in RL"

        # Categorical distribution over discrete actions
        action_dist = torch.distributions.Categorical(action_probs)

        # Sample action if not provided
        if action is None:
            action = action_dist.sample()

        # Log prob of selected action
        log_prob = action_dist.log_prob(action)

        # Entropy for exploration bonus
        entropy = action_dist.entropy()

        return action, log_prob, entropy, value


# ============================================================================
# EXPERIENCE COLLECTION AND STORAGE
# ============================================================================


class RolloutBuffer:
    """
    Buffer for storing trajectories (rollouts) collected from the environment.
    """

    def __init__(self, rollout_steps: int, num_actions: int, device: torch.device):
        self.rollout_steps = rollout_steps
        self.num_actions = num_actions
        self.device = device
        self.clear()

    def clear(self):
        """Reset the buffer for a new rollout."""
        self.states = []  # list of np arrays, each (210,160)
        self.actions = []  # list of ints
        self.log_probs = []  # list of tensors (scalar)
        self.rewards = []  # list of floats
        self.values = []  # list of floats (scalar values, not tensors)
        self.dones = []  # list of bools

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        """
        Add one step of experience to the buffer.
        """
        self.states.append(state)  # raw observation
        self.actions.append(action)  # int
        self.log_probs.append(log_prob.detach())  # detach from graph
        self.rewards.append(float(reward))  # float
        self.values.append(value.detach().item())  # store scalar value
        self.dones.append(done)  # bool

    def compute_advantages(
        self, next_value: torch.Tensor, gamma: float, gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        """
        # Convert to 1D tensors on correct device
        values = torch.tensor(
            self.values, dtype=torch.float32, device=self.device
        )  # (T,)
        rewards = torch.tensor(
            self.rewards, dtype=torch.float32, device=self.device
        )  # (T,)
        dones = torch.tensor(
            self.dones, dtype=torch.float32, device=self.device
        )  # (T,)

        # next_value is (1,1) or (1,) tensor → flatten to scalar
        next_value = next_value.view(-1)[0]
        values = torch.cat([values, next_value.unsqueeze(0)])  # (T+1,)

        advantages = []
        gae = 0.0

        # Iterate backwards over time steps
        for t in reversed(range(len(self.rewards))):
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_nonterminal - values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )  # (T,)
        returns = advantages + values[:-1]  # (T,)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


# ============================================================================
# PPO TRAINER
# ============================================================================


class PPOTrainer:
    """
    Main trainer for PPO algorithm:
    - Collect rollouts
    - Compute advantages
    - Update actor-critic
    """

    def __init__(
        self, env, config: Dict, device: torch.device = None, seed: int = None
    ):
        # Auto-select device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.env = env
        self.config = config
        self.seed = seed

        # Hyperparameters from config
        self.num_actions = env.action_space.n
        self.learning_rate = config["hyperparameters"]["learning_rate"]
        self.gamma = config["hyperparameters"]["gamma"]
        self.gae_lambda = config["hyperparameters"]["gae_lambda"]
        self.clip_ratio = config["hyperparameters"]["clip_ratio"]
        self.entropy_coeff = config["hyperparameters"]["entropy_coeff"]
        self.value_coeff = config["hyperparameters"]["value_coeff"]
        self.max_grad_norm = config["hyperparameters"]["max_grad_norm"]
        self.epochs_per_update = config["training"]["epochs_per_update"]
        self.batch_size = config["training"]["batch_size"]
        self.rollout_steps = config["training"]["rollout_steps"]

        # Actor-critic model
        self.actor_critic = PPOActorCritic(num_actions=self.num_actions).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.learning_rate
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            self.rollout_steps, self.num_actions, self.device
        )

        # Logging metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    # ---------- Helper to convert raw state to tensor ----------

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert raw env state (210x160 or 210x160x1) → (1,1,210,160) on device.
        """
        state_tensor = torch.from_numpy(state).float()
        if state_tensor.dim() == 3:  # (210,160,1)
            state_tensor = state_tensor.squeeze(-1)
        # Now (210,160) → add batch and channel dims → (1,1,210,160)
        state_tensor = state_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        return state_tensor

    # ---------- Rollout collection ----------

    def collect_rollout(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect one rollout (batch of experiences) from the environment.
        """
        # Initialize state if needed
        if not hasattr(self, "state") or self.env_done:
            if self.seed is not None:
                self.state, _ = self.env.reset(seed=self.seed)
            else:
                self.state, _ = self.env.reset()
            self.env_done = False

        # Clear previous rollout data
        self.rollout_buffer.clear()

        episode_reward = 0.0
        episode_length = 0

        for step in range(self.rollout_steps):
            # Convert current state to tensor (1,1,210,160)
            state_tensor = self._state_to_tensor(self.state)

            # Sample action from policy
            with torch.no_grad():
                action, log_prob, entropy, value = (
                    self.actor_critic.get_action_and_value(state_tensor)
                )

            # Convert action to numpy int for env
            action_np = int(action.cpu().numpy()[0])

            # Step the environment
            next_state, reward, terminated, truncated, info = self.env.step(action_np)

            done = terminated or truncated

            # Store in rollout buffer
            self.rollout_buffer.add(
                state=self.state,
                action=action_np,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
            )

            # Update state and episode stats
            self.state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1

                # Reset episode stats
                if self.seed is not None:
                    self.state, _ = self.env.reset(seed=self.seed)
                else:
                    self.state, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

        # Value of last state for GAE
        with torch.no_grad():
            state_tensor = self._state_to_tensor(self.state)
            _, _, _, next_value = self.actor_critic.get_action_and_value(state_tensor)

        # Compute advantages and returns
        advantages, returns = self.rollout_buffer.compute_advantages(
            next_value, self.gamma, self.gae_lambda
        )

        return advantages, returns

    # ---------- Policy update ----------

    def update_policy(self, advantages: torch.Tensor, returns: torch.Tensor):
        """
        Update policy and value networks using PPO clipped objective.
        """
        # Convert stored states → (T,1,210,160)
        states_list = []
        for state in self.rollout_buffer.states:
            st = torch.from_numpy(state).float()
            if st.dim() == 3:
                st = st.squeeze(-1)  # (210,160,1) → (210,160)
            st = st.unsqueeze(0).unsqueeze(0)  # (1,1,210,160)
            states_list.append(st)

        # (T,1,210,160)
        states = torch.cat(states_list, dim=0).to(self.device)

        # Actions and old log probs
        actions = torch.tensor(
            self.rollout_buffer.actions, dtype=torch.long, device=self.device
        )  # (T,)
        old_log_probs = torch.stack(self.rollout_buffer.log_probs).to(
            self.device
        )  # (T,)

        # Number of timesteps in rollout
        batch_size_total = states.size(0)
        indices = np.arange(batch_size_total)

        # Multiple epochs over same rollout
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)

            for start_idx in range(0, batch_size_total, self.batch_size):
                end_idx = min(start_idx + self.batch_size, batch_size_total)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]  # (B,1,210,160)
                batch_actions = actions[batch_indices]  # (B,)
                batch_advantages = advantages[batch_indices]  # (B,)
                batch_returns = returns[batch_indices]  # (B,)
                batch_old_log_probs = old_log_probs[batch_indices]  # (B,)

                # Forward pass through actor-critic
                action_logits, values = self.actor_critic.forward(
                    batch_states
                )  # values: (B,1)

                # New log probs
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy()

                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * batch_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Entropy bonus (negative because we maximize entropy)
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )

                self.optimizer.step()

    # ---------- Training loop ----------

    def train(self, total_episodes: int, log_interval: int = 100) -> Dict[str, List]:
        """
        Main training loop for PPO.
        """
        print(f"Starting PPO training for {total_episodes} episodes...")
        print(
            f"Config: LR={self.learning_rate}, Entropy={self.entropy_coeff}, "
            f"GAE-Lambda={self.gae_lambda}"
        )

        # Initial reset
        if self.seed is not None:
            self.state, _ = self.env.reset(seed=self.seed)
        else:
            self.state, _ = self.env.reset()
        self.env_done = False

        metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "moving_avg_reward": [],
        }

        while self.episode_count < total_episodes:
            # Collect rollout and compute advantages
            advantages, returns = self.collect_rollout()

            # Policy update
            self.update_policy(advantages, returns)

            # Logging
            if self.episode_count % log_interval == 0 and self.episode_count > 0:
                recent_rewards = self.episode_rewards[-log_interval:]
                avg_reward = np.mean(recent_rewards)
                max_reward = np.max(recent_rewards)
                min_reward = np.min(recent_rewards)

                print(
                    f"Episodes: {self.episode_count:4d} | "
                    f"Avg Reward: {avg_reward:7.2f} | "
                    f"Max: {max_reward:7.2f} | "
                    f"Min: {min_reward:7.2f}"
                )

                metrics["moving_avg_reward"].append(avg_reward)

            metrics["episode_rewards"] = self.episode_rewards.copy()
            metrics["episode_lengths"] = self.episode_lengths.copy()

        print("Training complete!")
        return metrics

    # ---------- Save / load ----------

    def save_model(self, path: str):
        torch.save(self.actor_critic.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for training.
    """

    import gymnasium as gym
    from gymnasium.wrappers import GrayscaleObservation

    # Load configuration from config.json
    with open("config.json", "r") as f:
        config = json.load(f)

    # Create Atari Pong environment from config["environment"], e.g. "ALE/Pong-v5"
    env = gym.make(config["environment"], render_mode=None)

    # Wrap with grayscale (keeps original 210x160 resolution)
    env = GrayscaleObservation(env)  # keep_dim=False → (210,160)

    # Create PPO trainer
    trainer = PPOTrainer(env, config)

    # Train
    metrics = trainer.train(
        total_episodes=config["training"]["total_episodes"], log_interval=100
    )

    # Create results directory if needed
    os.makedirs("results/logs", exist_ok=True)

    # Save model
    trainer.save_model("results/ppo_pong_model.pt")

    print("Training complete! Check results/ directory for logs and plots.")
