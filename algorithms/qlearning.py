"""
Q-Learning (Value-based, Off-policy, Tabular)

Description:
- A model-free, value-based algorithm that learns an action-value function (Q-function) to determine the 
  expected future rewards for taking an action in a given state. It updates Q-values iteratively using the Bellman equation.

Flow:
1) Epsilon-greedy action selection from a state-action value table.
2) After each step, compute TD target r + γ max_a' Q(s', a') and update the visited entry.
3) Decay epsilon to shift from exploration to exploitation over time.
"""

import random

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent for small discrete state spaces."""

    def __init__(self, state_shape: int, num_actions: int = 4, alpha: float = 0.1, gamma: float = 0.99) -> None:
        # Q-table stored as dict keyed by flattened state tuple -> action values
        self.q_table: dict[tuple, np.ndarray] = {}
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        self.state_shape = state_shape

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy pick: random with prob ε, else argmax Q."""
        key = tuple(state.tolist())
        if random.random() < self.epsilon or key not in self.q_table:
            return random.randrange(self.num_actions)
        return int(np.argmax(self.q_table[key]))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """TD(0) update for a single transition; returns the TD error magnitude."""
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
        # Gradually reduce exploration.
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return float(td_error)
