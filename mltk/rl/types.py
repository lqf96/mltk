from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional
from dataclasses import dataclass, field
from mltk.types.gym import O, A, R, _AbstractEnv

from collections import defaultdict

from mltk.engine import State

if TYPE_CHECKING:
    from .agent import RLAgentBase

__all__ = [
    "RLState",
    "Transition"
]

@dataclass(eq=False)
class RLState(State, Generic[O, A, R]):
    agent: RLAgentBase[O, A, R]
    env: _AbstractEnv[O, A, R]

    training: bool = False
    metrics: dict[str, Any] = field(default_factory=lambda: defaultdict(lambda: 0.))
    transition: Optional[Transition] = None

@dataclass(eq=False, frozen=True)
class Transition(Generic[O, A, R]):
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
    extra: dict[str, Any]
