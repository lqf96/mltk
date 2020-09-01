from typing import Generic
from mltk.types.gym import A

import numpy as np
import torch as th
from torch.distributions import Distribution

import mltk.util as mu

__all__ = [
    "Policy",
    "DeterministicPolicy",
    "StochasticPolicy"
]

class Policy(Generic[A]):
    def sample(self) -> A:
        raise NotImplementedError

class DeterministicPolicy(Policy[np.ndarray]):
    __slots__ = ("actions",)

    def __init__(self, actions: th.Tensor):
        self.actions = actions
    
    def sample(self) -> np.ndarray:
        return self.actions.cpu().numpy()

class StochasticPolicy(Policy[np.ndarray]):
    __slots__ = ("dist", "rand")

    def __init__(self, dist: Distribution, rand: th.Generator = th.default_generator):
        self.dist = dist
        self.rand = rand
    
    def sample(self) -> np.ndarray:
        with mu.use_rand(self.rand):
            actions: th.Tensor = self.dist.sample(()).cpu() # type: ignore
        # Convert actions to scalar or NumPy array
        return actions.item() if mu.isscalar(actions) else actions.numpy()
