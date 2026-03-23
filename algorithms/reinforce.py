"""
Policy Gradient (REINFORCE) (Policy-based)

Description:
- Description: A policy-based method that directly optimizes the policy by adjusting its parameters using 
  gradient ascent on expected rewards. It’s simple but can suffer from high variance.

Flow:
1) Roll out an episode with a stochastic policy πθ.
2) Compute returns G_t from the collected rewards.
3) Update θ in the direction of ∑_t log πθ(a_t|s_t) * (G_t - baseline) (baseline omitted here).
"""

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common import to_tensor


class PolicyNetwork(nn.Module):
    """Simple MLP policy that outputs logits over discrete actions."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReinforceAgent:
    """Vanilla REINFORCE agent using episodic returns for updates."""

    def __init__(self, state_dim: int, num_actions: int = 4, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.policy = PolicyNetwork(state_dim, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Sample an action from the policy distribution; return action and log-prob."""
        logits = self.policy(state.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def update_policy(self, log_probs: list[torch.Tensor], rewards: list[float]) -> float:
        """Compute returns, normalize them, and apply one policy gradient step."""
        returns = []
        g = 0.0
        # Walk backward to build discounted returns.
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        # Normalize to reduce variance.
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        loss_terms = []
        for log_p, ret in zip(log_probs, returns_tensor):
            loss_terms.append(-log_p * ret)
        loss = torch.stack(loss_terms).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
