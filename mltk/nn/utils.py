from __future__ import annotations

import copy

import torch as th
from torch import nn

__all__ = [
    "Concat",
    "RepeatLayers"
]

class Concat(nn.Module):
    __slots__ = ("dim",)

    def __init__(self, dim: int = -1):
        super().__init__()
        
        self.dim = dim
    
    def forward(self, inputs: tuple[th.Tensor, ...]) -> th.Tensor:
        return th.cat(inputs, self.dim)

class RepeatLayers(nn.Sequential):
    def __init__(self, layers, *args, times: int):
        all_layers = []
        
        for _ in range(times):
            all_layers.extend(layers())

        super().__init__(*all_layers)
