from typing import Tuple

import torch as th
from torch.nn import Module

__all__ = [
    "Concat"
]

class Concat(Module):
    __slots__ = ("dim",)

    def __init__(self, dim: int = -1):
        self.dim = dim
    
    def forward(self, inputs: Tuple[th.Tensor, ...]):
        return th.cat(inputs, self.dim)
