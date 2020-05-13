from typing import Any, Protocol, SupportsIndex, TypeVar, Union
from torch.types import _size

import torch as th
from torch.nn import Module

__all__ = [
    "ModuleT",
    "Shape",
    "Device",
    "TensorLike",
    "TensorT"
]

# A shape-like type allowing a single integer to be used as a shape
Shape = Union[SupportsIndex, _size]
# A device-like type
Device = Union[th.device, str, int]

ModuleT = TypeVar("ModuleT", bound=Module)

class TensorLike(Protocol):
    def __add__(self: "TensorT", other: Any) -> "TensorT": ...
    def __radd__(self: "TensorT", other: Any) -> "TensorT": ...
    def __sub__(self: "TensorT", other: Any) -> "TensorT": ...
    def __rsub__(self: "TensorT", other: Any) -> "TensorT": ...
    def __mul__(self: "TensorT", other: Any) -> "TensorT": ...
    def __rmul__(self: "TensorT", other: Any) -> "TensorT": ...
    def __truediv__(self: "TensorT", other: Any) -> "TensorT": ...
    def __rtruediv__(self: "TensorT", other: Any) -> "TensorT": ...

TensorT = TypeVar("TensorT", bound=TensorLike)