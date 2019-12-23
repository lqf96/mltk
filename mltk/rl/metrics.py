from typing import Any, Optional, Generator

from enum import Enum

from ignite.engine import Events
from ignite.metrics import Metric

from mltk.metrics import GeneratorMetric, OneshotMetric
from mltk import util

__all__ = [
    "episode_length",
    "episode_reward"
]

@GeneratorMetric.wraps()
def episode_length() -> int:
    count = 0
    # Episode loop
    while True:
        output = yield
        # End of episode
        if output is None:
            break
        count += 1
    return count

@GeneratorMetric.wraps()
def episode_reward(discount_factor: float = 1):
    total_reward = 0
    # Episode loop
    while True:
        output = yield
        # End of episode
        if output is None:
            break
        # Update total reward
        total_reward *= discount_factor
        total_reward += output.transition.reward
    return total_reward
