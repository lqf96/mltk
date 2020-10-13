from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import copy

import torch as th
from torch import nn

__all__ = [
    "Concat",
    "mlp"
]

class Concat(nn.Module):
    __slots__ = ("dim",)

    def __init__(self, dim: int = -1):
        super().__init__()
        
        self.dim = dim
    
    def forward(self, inputs: Tuple[th.Tensor, ...]) -> th.Tensor:
        return th.cat(inputs, self.dim)

def mlp():
    pass
