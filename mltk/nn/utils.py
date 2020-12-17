from __future__ import annotations

from typing import Optional, Union
from mltk.types import Shape

import torch as th
from torch import nn

import mltk.util as mu

__all__ = [
    "Concat",
    "Cropping2D",
    "Lambda",
    "RepeatLayers",
    "Reshape"
]

class Concat(nn.Module):
    __slots__ = ("dim",)

    def __init__(self, dim: int = -1):
        super().__init__()
        
        self.dim = dim
    
    def forward(self, inputs: tuple[th.Tensor, ...]) -> th.Tensor:
        return th.cat(inputs, self.dim)

# TODO: Replace this with a universal `Cropping`
class Cropping2D(nn.Module):
    def __init__(self, cropping: Union[int, tuple[int, ...]]):
        super().__init__()

        if isinstance(cropping, int):
            cropping = (cropping,)*4
        
        self._crop_h = slice(cropping[2], -cropping[3] or None)
        self._crop_w = slice(cropping[0], -cropping[1] or None)
    
    def forward(self, inputs: th.Tensor) -> th.Tensor:
        return inputs[..., self._crop_h, self._crop_w]

class Lambda(nn.Module):
    __slots__ = ("f",)

    def __init__(self, f):
        super().__init__()

        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)

class RepeatLayers(nn.Sequential):
    def __init__(self, layers, *args, times: int):
        all_layers: list[nn.Module] = []
        
        for _ in range(times):
            all_layers.extend(layers())

        super().__init__(*all_layers)

class Reshape(nn.Module):
    __slots__ = ("shape", "start_dim", "end_dim")

    def __init__(self, shape: Shape, start_dim: int = 1, end_dim: int = -1):
        super().__init__()

        self.shape = mu.as_size(shape)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        inputs_shape = inputs.shape
        inputs_ndim = len(inputs_shape)

        # Normalize start and end dimensions
        start_dim = self.start_dim%inputs_ndim
        end_dim = self.end_dim%inputs_ndim
        # Compute outputs shape
        outputs_shape = inputs_shape[:start_dim]+self.shape+inputs_shape[end_dim+1:]

        return inputs.reshape(outputs_shape)
