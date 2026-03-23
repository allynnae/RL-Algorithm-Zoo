"""
Deep Q-Network (DQN) (Value-based, Approximate)

Description:
- Description: An extension of Q-Learning that uses a neural network to approximate the Q-function, enabling it 
  to handle high-dimensional state spaces like images. It employs experience replay and target networks for stability.
  
Flow:
1) Epsilon-greedy action selection over Q-network outputs.
2) Store transitions in replay buffer; sample batches to decorrelate updates.
3) Compute TD target r + γ max_a' Q_target(s', a') and minimize MSE loss against Q_online(s, a).
4) Periodically copy online weights to target network for stability.
"""

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common import Transition, to_tensor


class QNetwork(nn.Module):
    """Three-layer MLP approximating Q(s, a)."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 20000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.999,
        target_update: int = 50,
        warmup_min: int = 200,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps_done = 0
        self.warmup_min = warmup_min

        self.q_net = QNetwork(state_dim, num_actions)
        self.target_net = QNetwork(state_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer: deque[Transition] = deque(maxlen=buffer_size)

    def select_action(self, state: torch.Tensor) -> int:
        """Epsilon-greedy action selection from Q-network predictions."""
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_vals = self.q_net(state.unsqueeze(0))
                action = int(torch.argmax(q_vals, dim=1).item())
        # Decay epsilon after every decision.
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        return action

    def store(self, transition: Transition) -> None:
        """Add a single transition to replay."""
        self.buffer.append(transition)

    def train_step(self) -> float:
        """Sample a batch from replay and run one gradient step; returns loss."""
        if len(self.buffer) < max(self.batch_size, self.warmup_min):
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        states = torch.stack([to_tensor(tr.state) for tr in batch])
        actions = torch.tensor([tr.action for tr in batch], dtype=torch.long)
        rewards = torch.tensor([tr.reward for tr in batch], dtype=torch.float32)
        next_states = torch.stack([to_tensor(tr.next_state) for tr in batch])
        dones = torch.tensor([tr.done for tr in batch], dtype=torch.float32)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


def warmup_replay(env, agent: DQNAgent, steps: int = 500) -> None:
    """Collect random experience to prefill replay buffer before learning."""
    render_prev = env.render_enabled
    env.render_enabled = False
    state = env.reset()
    for _ in range(steps):
        action = random.randrange(agent.num_actions)
        next_state, reward, done, _ = env.step(action)
        agent.store(Transition(state, action, reward, next_state, done))
        state = env.reset() if done else next_state
    env.render_enabled = render_prev
