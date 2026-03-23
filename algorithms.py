"""
Algorithm implementations: Q-Learning, REINFORCE, A2C, Decision Transformer, plus shared helpers.
"""

import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Shared transition tuple
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim: int, num_actions: int = 4, d_model: int = 64, nhead: int = 4, seq_len: int = 10) -> None:
        super().__init__()
        self.state_embed = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.action_head = nn.Linear(d_model, num_actions)
        self.seq_len = seq_len

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.state_embed(states)
        x = self.pos_encoding(x)
        encoded = self.transformer(x)
        last_token = encoded[:, -1, :]
        return self.action_head(last_token)


class DecisionTransformerAgent:
    def __init__(self, state_dim: int, num_actions: int = 4, seq_len: int = 10, lr: float = 1e-3) -> None:
        self.seq_len = seq_len
        self.model = DecisionTransformer(state_dim, num_actions, seq_len=seq_len)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.buffer: deque[Transition] = deque(maxlen=5000)

    def store(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample_batch(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.buffer) < self.seq_len:
            return None, None
        sequences_states = []
        sequences_actions = []
        for _ in range(batch_size):
            start = random.randint(0, len(self.buffer) - self.seq_len)
            seq = list(self.buffer)[start : start + self.seq_len]
            state_stack = [to_tensor(tr.state) for tr in seq]
            action_target = seq[-1].action
            sequences_states.append(torch.stack(state_stack))
            sequences_actions.append(action_target)
        states_tensor = torch.stack(sequences_states)
        actions_tensor = torch.tensor(sequences_actions, dtype=torch.long)
        return states_tensor, actions_tensor

    def train_step(self, batch_size: int = 32) -> float:
        states, actions = self.sample_batch(batch_size)
        if states is None:
            return 0.0
        logits = self.model(states)
        loss = self.loss_fn(logits, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def select_action(self, recent_states: deque) -> int:
        if len(recent_states) < self.seq_len:
            return random.randrange(4)
        state_window = list(recent_states)[-self.seq_len :]
        state_tensor = torch.stack([to_tensor(s) for s in state_window]).unsqueeze(0)
        logits = self.model(state_tensor)
        action = torch.argmax(logits, dim=1)
        return int(action.item())


# Fill the replay buffer with random episodes before DT training.
def warmup_buffer(env, agent: DecisionTransformerAgent, episodes: int = 50) -> None:
    original_render = env.render_enabled
    env.render_enabled = False
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = random.randrange(4)
            next_state, reward, done, _ = env.step(action)
            agent.store(Transition(state, action, reward, next_state, done))
            state = next_state
    env.render_enabled = original_render
