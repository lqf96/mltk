from typing import Any, Optional, Union
from abc import ABC, abstractmethod

import torch as th
from torch.distributions import Distribution
from gym import Env
from gym.spaces import Discrete

import mltk.util as mu
from .execution import Step, Transition

__all__ = [
    "RLAgent"
]

class RLAgent(ABC):
    def __init__(self, env: Env, rand: th.Generator = th.default_generator):
        ## Environment
        self.env = env
        ## Random number generator
        self.rand = rand

    @abstractmethod
    def policy(self, observations: th.Tensor, training_step: Optional[Step] = None,
        **kwargs: Any) -> Union[Distribution, th.Tensor]:
        raise NotImplementedError

    def fit(self, step: Step, transition: Transition, **kwargs):
        raise NotImplementedError

    def update(self, step: Step, **kwargs):
        raise NotImplementedError

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def act(self, observation: th.Tensor, **kwargs: Any) -> th.Tensor:
        action = self.policy(observation, **kwargs)
        # Sample action from policy distribution
        if isinstance(action, Distribution):
            with mu.use_rand(self.rand):
                action = action.sample()
        return action

    @property
    def n_states(self) -> int:
        """ Return number of states in the discrete state space. """
        state_space = self.env.observation_space
        # State space must be discrete
        if not isinstance(state_space, Discrete):
            raise TypeError
        return state_space.n

    @property
    def n_actions(self) -> int:
        """ Return number of actions in the discrete action space. """
        action_space = self.env.action_space
        # Action space must be discrete
        if not isinstance(action_space, Discrete):
            raise TypeError
        return action_space.n
