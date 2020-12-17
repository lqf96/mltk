from __future__ import annotations

from typing import Protocol, SupportsIndex, TypeVar, Union
from torch.types import _size

import torch as th
from torch.nn import Module

__all__ = [
    "ActivationFactory",
    "ActivationFunc",
    "Device",
    "ModuleT",
    "Numerical",
    "NumericalT",
    "Shape",
    "TensorLike",
    "TensorT"
]

# A shape-like type allowing a single integer to be used as a shape
Shape = Union[SupportsIndex, _size]
# A device-like type
Device = Union[th.device, str, int]

# Generic type variable for subclasses of PyTorch module
ModuleT = TypeVar("ModuleT", bound=Module)

# A number-like or tensor-like type
Numerical = Union["TensorLike", int, float]
NumericalT = TypeVar("NumericalT", bound=Numerical)
# Generic type variable for tensor-like types
TensorT = TypeVar("TensorT", bound="TensorLike")

class TensorLike(Protocol):
    def __add__(self: TensorT, other: Numerical) -> TensorT: ...
    def __radd__(self: TensorT, other: Numerical) -> TensorT: ...
    def __sub__(self: TensorT, other: Numerical) -> TensorT: ...
    def __rsub__(self: TensorT, other: Numerical) -> TensorT: ...
    def __mul__(self: TensorT, other: Numerical) -> TensorT: ...
    def __rmul__(self: TensorT, other: Numerical) -> TensorT: ...
    def __truediv__(self: TensorT, other: Numerical) -> TensorT: ...
    def __rtruediv__(self: TensorT, other: Numerical) -> TensorT: ...
    def __pow__(self: TensorT, other: Numerical) -> TensorT: ...
    def __rpow__(self: TensorT, other: Numerical) -> TensorT: ...

class ActivationFunc(Protocol):
    def __call__(self, inputs: th.Tensor, inplace: bool = ...) -> th.Tensor: ...

class ActivationFactory(Protocol):
    def __call__(self, inplace: bool = ...) -> Module: ...
