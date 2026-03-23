"""
Shared utilities for RL agents.

Provides:
- Transition tuple for replay buffers
- Helper to convert numpy states to torch tensors
"""

from collections import namedtuple

import numpy as np
import torch

# Standard transition container used by value-based agents with replay.
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def to_tensor(state: np.ndarray) -> torch.Tensor:
    """Convert a numpy state vector to a float32 torch tensor."""
    return torch.tensor(state, dtype=torch.float32)
