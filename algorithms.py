"""
Algorithm implementations: Q-Learning, REINFORCE, A2C, and DQN, plus shared helpers.
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Shared transition tuple for replay-style buffers.
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

# Convert numpy state vector to a torch tensor.
def to_tensor(state: np.ndarray) -> torch.Tensor:
    return torch.tensor(state, dtype=torch.float32)


class PolicyNetwork(nn.Module):
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
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QLearningAgent:
    def __init__(self, state_shape: int, num_actions: int = 4, alpha: float = 0.1, gamma: float = 0.99) -> None:
        self.q_table = {}
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        self.state_shape = state_shape

    def select_action(self, state: np.ndarray) -> int:
        key = tuple(state.tolist())
        if random.random() < self.epsilon or key not in self.q_table:
            return random.randrange(self.num_actions)
        return int(np.argmax(self.q_table[key]))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        key = tuple(state.tolist())
        next_key = tuple(next_state.tolist())
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.num_actions, dtype=np.float32)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.num_actions, dtype=np.float32)
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_key])
        td_error = target - self.q_table[key][action]
        self.q_table[key][action] += self.alpha * td_error
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return float(td_error)


class ReinforceAgent:
    def __init__(self, state_dim: int, num_actions: int = 4, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.policy = PolicyNetwork(state_dim, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits = self.policy(state.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def update_policy(self, log_probs: list[torch.Tensor], rewards: list[float]) -> float:
        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        loss_terms = []
        for log_p, ret in zip(log_probs, returns_tensor):
            loss_terms.append(-log_p * ret)
        loss = torch.stack(loss_terms).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class A2CAgent:
    def __init__(self, state_dim: int, num_actions: int = 4, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.policy = PolicyNetwork(state_dim, num_actions)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.gamma = gamma

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        logits = self.policy(state.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(state.unsqueeze(0))
        return int(action.item()), log_prob, value

    def update(self, log_prob: torch.Tensor, value: torch.Tensor, reward: float, next_value: torch.Tensor, done: bool) -> float:
        target = reward + (0.0 if done else self.gamma * next_value.item())
        advantage = target - value.item()
        actor_loss = -log_prob * advantage
        critic_loss = (value - target) ** 2
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


# ----- DQN components -----
class QNetwork(nn.Module):
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
        epsilon_decay: float = 0.999,  # slower decay to keep early exploration
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
        # Epsilon-greedy over Q-network predictions.
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_vals = self.q_net(state.unsqueeze(0))
                action = int(torch.argmax(q_vals, dim=1).item())
        # Decay epsilon after each decision.
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        return action

    def store(self, transition: Transition) -> None:
        self.buffer.append(transition)

# Prefill the replay buffer with random experience to avoid a cold start.
def warmup_replay(env, agent: DQNAgent, steps: int = 500) -> None:
    render_prev = env.render_enabled
    env.render_enabled = False
    state = env.reset()
    for _ in range(steps):
        action = random.randrange(agent.num_actions)
        next_state, reward, done, _ = env.step(action)
        agent.store(Transition(state, action, reward, next_state, done))
        state = env.reset() if done else next_state
    env.render_enabled = render_prev
    def train_step(self) -> float:
        # Need enough samples to form a batch.
        if len(self.buffer) < max(self.batch_size, self.warmup_min):
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        states = torch.stack([to_tensor(tr.state) for tr in batch])
        actions = torch.tensor([tr.action for tr in batch], dtype=torch.long)
        rewards = torch.tensor([tr.reward for tr in batch], dtype=torch.float32)
        next_states = torch.stack([to_tensor(tr.next_state) for tr in batch])
        dones = torch.tensor([tr.done for tr in batch], dtype=torch.float32)

        # Current Q estimates for taken actions.
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Target Q using the frozen target network.
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically sync target network for stability.
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
