from __future__ import annotations

from typing import Optional
from collections.abc import Iterator
from mltk.types import Device, Shape

from itertools import repeat

import torch as th
from torch import Tensor

import mltk.util as mu

class SymbolicTensor(Tensor):
    def __new__(cls, shape: Shape, *, dtype: Optional[th.dtype] = None, device: Device = "cpu"):
        size = mu.as_size(shape)
        dtype = dtype or th.get_default_dtype()
        device = th.device(device)

        tensor = super().__new__(cls, 0, *size, device=device)
        return tensor.to(dtype)

    def _iter_impl(self) -> Iterator[SymbolicTensor]:
        try:
            length, *elem_shape = self.size()
        except ValueError:
            raise TypeError("cannot iterate over 0-d structure")
        
        elem = self.__class__(elem_shape, device=self.device)
        return repeat(elem, length)

    def __len__(self) -> int:
        shape = self.size()

        if len(shape)==0:
            raise TypeError("length unavailable for 0-d structure")
        else:
            return shape[0]
    
    def __getitem__(self, idx) -> SymbolicTensor:
        # Prepend a dummy slice-all dimension for the symbolic tensor
        base_idx = (mu.SLICE_ALL, *idx) if isinstance(idx, tuple) else (mu.SLICE_ALL, idx)
        
        return super().__getitem__(base_idx)

    def __repr__(self) -> str:
        cls = self.__class__
        device = self.device

        repr_str = f"{cls.__name__}(shape={tuple(self.shape)}"
        # Show dtype
        if self.dtype!=th.get_default_dtype():
            repr_str += f", dtype={self.dtype}"
        # Show device for non-CPU tensor
        if device.type!="cpu":
            repr_str += f", device='{device.type}:{device.index}'"
        repr_str += ")"

        return repr_str

    def size(self) -> th.Size:
        return super().shape[1:]
    
    def dim(self) -> int:
        return super().ndim-1

    # Shared iteration implementation
    __iter__ = _iter_impl
    __reversed__ = _iter_impl
    # Alias for `size` method
    shape = property(size)
    # Alias for `dim` method
    ndimension = dim
    ndim = property(dim)
