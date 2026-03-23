"""
Advantage Actor-Critic (A2C) (Hybrid)

Description:
- Description: Combines policy-based (actor) and value-based (critic) methods. The actor updates the policy, 
  while the critic estimates the value function to reduce variance in gradient updates.

Flow:
1) Actor picks actions from policy πθ; critic estimates V(s).
2) For each step, compute TD target r + γ V(s') and advantage A = target - V(s).
3) Update actor with policy gradient weighted by A; update critic to regress toward target.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common import to_tensor


class PolicyNetwork(nn.Module):
    """Actor network producing logits over actions."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """Critic network estimating state value V(s)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class A2CAgent:
    """Synchronous actor-critic agent with shared optimizer."""

    def __init__(self, state_dim: int, num_actions: int = 4, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.policy = PolicyNetwork(state_dim, num_actions)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.gamma = gamma

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action, return log-prob and value estimate for update."""
        logits = self.policy(state.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(state.unsqueeze(0))
        return int(action.item()), log_prob, value

    def update(self, log_prob: torch.Tensor, value: torch.Tensor, reward: float, next_value: torch.Tensor, done: bool) -> float:
        """One-step advantage update for actor and critic."""
        target = reward + (0.0 if done else self.gamma * next_value.item())
        advantage = target - value.item()
        actor_loss = -log_prob * advantage
        critic_loss = (value - target) ** 2
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
