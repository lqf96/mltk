from __future__ import annotations

from typing import Any, Generic, Optional
from collections.abc import Mapping
from mltk.types import Device
from mltk.types.gym import O, A, R, Discrete, Space, _AbstractEnv

import torch as th
from torch.distributions import Distribution
from torch import nn

import mltk.util as mu
from .types import RLState

__all__ = [
    "RLAgentBase",
    "RLAgent"
]

class RLAgentBase(nn.Module, Generic[O, A, R]):
    def __init__(self, env: _AbstractEnv[O, A, R], rand: th.Generator = th.default_generator):
        super().__init__()

        self.env = env
        self.rand_cpu = mu.derive_rand(rand, "cpu")

    @property
    def device(self) -> th.device:
        param = next(iter(self.parameters()))
        return param.device

    @property
    def dtype(self) -> th.dtype:
        param = next(iter(self.parameters()))
        return param.dtype

    @property
    def observation_space(self) -> "Space[O]":
        return self.env.observation_space

    @property
    def action_space(self) -> "Space[A]":
        return self.env.action_space

    def observe(self, state: RLState[O, A, R], observation: O, done: bool) -> None:
        pass

    def act(self, state: RLState[O, A, R]) -> A:
        action = self(state)
        if isinstance(action, Distribution):
            action = action.sample()
        if isinstance(action, th.Tensor):
            action = action.detach().cpu()
            action = action.item() if mu.isscalar(action) else action.numpy()
        return action

    def train(self, state: RLState[O, A, R]) -> Optional[dict[str, Any]]:
        raise NotImplementedError

class RLAgent(RLAgentBase[O, A, float]):
    __slots__ = ()

    @property
    def n_obs(self) -> int:
        """ Number of observations in the discrete observations space. """
        obs_space = self.observation_space

        # Observation space must be discrete
        if not isinstance(obs_space, Discrete):
            #raise TypeError("`n_states` can only be used on discrete state spaces")
            return NotImplemented

        return obs_space.n

    @property
    def n_actions(self) -> int:
        """ Number of actions in the discrete action space. """
        action_space = self.action_space

        # Action space must be discrete
        if not isinstance(action_space, Discrete):
            #raise TypeError("`n_actions` can only be used on discrete action spaces")
            return NotImplemented

        return action_space.n
