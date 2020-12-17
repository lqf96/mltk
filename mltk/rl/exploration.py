from __future__ import annotations

from typing import Optional, Union

import torch as th
import torch.distributions as dists

import mltk.util as mu
from mltk.optim.decay import DecayFunc, apply_decay

__all__ = [
    "epsilon_greedy"
]

_DiscretePolicy = Union[
    dists.Categorical,
    dists.OneHotCategorical,
    dists.RelaxedOneHotCategorical
]

def epsilon_greedy(epsilon: float = 0.1, decay_func: Optional[DecayFunc] = None):
    def add_exploration(policy: _DiscretePolicy, n_steps: int):
        # Compute epsilon with optional decay
        eps = apply_decay(epsilon, decay_func, n_steps)

        # Compute policy with exploration
        new_policy_probs = (1-eps)*policy.probs+eps/policy.probs.shape[-1]
        return policy.__class__(probs=new_policy_probs)
    return add_exploration
