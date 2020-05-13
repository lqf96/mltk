from typing import Any, Generic, Optional
from mltk.types import Device
from mltk.types.gym import O, A, R, Discrete, Space, _AbstractEnv

from abc import ABC, abstractmethod

import torch as th

import mltk.util as mu
from .policy import Policy
from .types import Step, Transition

__all__ = [
    "RLAgent"
]

class _AbstractRLAgent(ABC, Generic[O, A, R]):
    __slots__ = ("env", "dtype", "device", "rand_cpu", "rand_dev")

    def __init__(self, env: _AbstractEnv[O, A, R], dtype: Optional[th.dtype] = None,
        device: Device = "cpu", rand: th.Generator = th.default_generator):
        self.env = env
        self.dtype = th.get_default_dtype() if dtype is None else dtype
        self.device = device = mu.as_device(device)
        self.rand_cpu = mu.derive_rand(rand, "cpu")
        self.rand_dev = mu.derive_rand(rand, device)
    
    @abstractmethod
    def policy(self, observations: O, *, training_step: Optional[Step] = None,
        **kwargs: Any) -> Policy[A]:
        raise NotImplementedError

    @property
    def observation_space(self) -> "Space[O]":
        return self.env.observation_space

    @property
    def action_space(self) -> "Space[A]":
        return self.env.action_space

    def act(self, observation: O, **kwargs: Any) -> A:
        return self.policy(observation, **kwargs).sample()

    def update_experiences(self, step: Step, transition: Transition) -> None:
        pass

    def update_policy(self, step: Step) -> None:
        raise NotImplementedError

class RLAgent(_AbstractRLAgent[O, A, float]):
    __slots__ = ()

    @property
    def n_obs(self) -> int:
        """ Number of observations in the discrete observations space. """
        obs_space = self.observation_space

        # Observation space must be discrete
        if not isinstance(obs_space, Discrete):
            raise TypeError("`n_states` can only be used on discrete state spaces")

        return obs_space.n

    @property
    def n_actions(self) -> int:
        """ Number of actions in the discrete action space. """
        action_space = self.action_space

        # Action space must be discrete
        if not isinstance(action_space, Discrete):
            raise TypeError("`n_actions` can only be used on discrete action spaces")

        return action_space.n
