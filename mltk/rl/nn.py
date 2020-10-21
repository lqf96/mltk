from typing import Optional, Protocol
from abc import abstractmethod

import torch as th
import torch.distributions as thd
from torch import nn

__all__ = [
    "DiscreteQNet",
    "QNet"
]
    
class DuelingQNet(nn.Module):
    def __init__(self, base_net: nn.Module, feat_dims: int, n_actions: int):
        super().__init__()
        self.base_net = base_net

        self._value_head = nn.Sequential(
            nn.Linear(feat_dims, feat_dims),
            nn.ELU(),
            nn.Linear(feat_dims, 1)
        )
        self._adv_head = nn.Sequential(
            nn.Linear(feat_dims, feat_dims),
            nn.ELU(),
            nn.Linear(feat_dims, n_actions)
        )
    
    def forward(self, states: th.Tensor) -> th.Tensor:
        head_inputs = self.base_net(states)
        values = self._value_head(head_inputs)
        advantages = self._adv_head(head_inputs)
        return values+advantages-advantages.mean(-1, keepdim=True)
