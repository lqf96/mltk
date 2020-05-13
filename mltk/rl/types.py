from typing import NamedTuple, Protocol
from mltk.types import StrDict
from mltk.types.gym import O, A, R

import numpy as np
import torch as th

__all__ = [
    "Step",
    "Transition"
]

class Step(NamedTuple):
    """\
    Information of the current step in RL training or execution.
    
    Attributes:
        episodes: Number of episodes.
        iterations: Number of iterations.
        episode_iterations: Number of iterations within the current episode.
    """
    episodes: int
    iterations: int
    episode_iterations: int

class Transition(NamedTuple):
    """\
    Information of the transition to the next state in an RL environment.
    
    Attributes:
        observation: Previous observation(s) of the environment.
        action: Action(s) of the agent(s).
        reward: Reward(s) received by the agent(s).
        next_observation: Next observation(s) of the environment.
        done: Indicates whether the current episode ends or not.
        extra: Extra information from the environment.
    """
    observation: O
    action: A
    reward: R
    next_observation: O
    done: bool
    extra: StrDict
