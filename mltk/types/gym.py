from typing import TYPE_CHECKING, Protocol, Tuple, TypeVar

import numpy as np

from .core import StrDict, T

__all__ = [
    "O",
    "A",
    "R",
    "Space",
    "Box",
    "StepResult",
    "MAReward",
    "Env",
    "MAEnv"
]

# Observation type
O = TypeVar("O")
# Action type
A = TypeVar("A")
# Reward type
R = TypeVar("R")

if TYPE_CHECKING:
    class Space(Protocol[T]):
        @property
        def dtype(self) -> np.dtype: ...

        @property
        def shape(self) -> Tuple[int, ...]: ...

        def contains(self, x: T) -> bool: ...

        def __contains__(self, x: T) -> bool: ...

    class Discrete(Space[int]):
        n: int
else:
    from gym.spaces import Discrete, Space

# Type of result returned by `env.step`
StepResult = Tuple[O, R, bool, StrDict]

class _AbstractEnv(Protocol[O, A, R]):
    @property
    def observation_space(self) -> "Space[O]": ...

    @property
    def action_space(self) -> "Space[A]": ...

    def step(self, action: A) -> StepResult[O, R]: ...

    def reset(self) -> O: ...

    def render(self, mode: str = ...) -> None: ...

    def close(self) -> None: ...

    @property
    def unwrapped(self) -> "_AbstractEnv[O, A, R]": ...

if TYPE_CHECKING:
    class Env(_AbstractEnv[O, A, float]):
        pass
else:
    from gym import Env
