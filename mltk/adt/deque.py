from typing import Optional, Any, Iterator
from mltk.types import Device, Shape

import torch as th

import mltk.util as mu

__all__ = [
    "Deque"
]

class Deque():
    ## Minimal capacity of a deque
    _MIN_CAPACITY = 4

    def __init__(self, shape: Shape, dtype: Optional[th.dtype] = None,
        device: Device = "cpu", max_len: Optional[int] = None):
        size = mu.as_size(shape)
        dtype = th.get_default_dtype() if dtype is None else dtype
        device = mu.as_device(device)

        self._buf = th.empty((self._MIN_CAPACITY, *size), dtype=dtype, device=device)
        self._begin = 0
        self._end = 0
        self._max_len = max_len

    def _expand(self, extra_capacity: int, expand_back: bool):
        # Valid data (from old buffer) and its length
        data = self.view()

        # Re-allocate buffer with new capacity
        self._buf = data.new_empty((
            self.capacity+extra_capacity, *data.shape[1:]
        ))
        # Adjust begin and end index
        if expand_back:
            self._begin += extra_capacity
            self._end += extra_capacity
        # Copy valid data to new buffer
        self.view()[:] = data

    # Deque is not hashable
    __hash__ = None

    def __len__(self) -> int:
        return self._end-self._begin
    
    def __getitem__(self, index: Any) -> th.Tensor:
        return self.view().__getitem__(index)
    
    def __setitem__(self, index, values):
        self.view().__setitem__(index, values)

    def __iter__(self) -> Iterator[th.Tensor]:
        return self.view().__iter__()
    
    def __bool__(self) -> bool:
        return self._end!=self._begin

    @property
    def capacity(self) -> int:
        """ Capacity of the deque. """
        return len(self._buf)

    @property
    def max_len(self) -> Optional[int]:
        """ Maximum size of the deque. None if unbounded. """
        return self._max_len

    @property
    def shape(self) -> th.Size:
        return self.view().shape
    
    @property
    def device(self) -> th.device:
        return self._buf.device

    def append(self, new_elem):
        # Convert new element to PyTorch tensor
        new_elem = th.as_tensor(new_elem)
        # Lazily initialize buffer
        if self._buf is None:
            self._buf = new_elem.new_empty((self._MIN_CAPACITY, *new_elem.shape))

        capacity = self.capacity
        # Expand the deque if it is full
        if self._end==capacity:
            self._expand(int(0.5*capacity), False)
        
        # Store new element
        self._buf[self._end] = new_elem
        # Update end index
        self._end += 1

        max_len = self._max_len
        # Remove the first element
        if max_len!=None and self.__len__()>max_len:
            self.pop_back()

    def pop_back(self) -> th.Tensor:
        # Empty deque check
        if self.__len__()==0:
            raise IndexError("attempt to pop from empty deque")

        # Get the first element
        pop_elem = self._buf[self._begin].clone()
        # Update begin index
        self._begin += 1

        # Shrink the deque
        if self.__len__()<=0.5*self.capacity:
            self.shrink_to_fit()
        
        return pop_elem

    def shrink_to_fit(self):
        # Valid data (from old buffer) and its length
        data = self.view()
        data_len = len(data)

        # New buffer capacity after shrinking
        new_capacity = max(data_len, self._MIN_CAPACITY)
        # Re-allocate buffer
        self._buf = data.new_empty((new_capacity, *data.shape[1:]))

        # Reset begin and end index
        self._begin = 0
        self._end = data_len
        # Copy valid data to new buffer
        self.view()[:] = data

    def view(self) -> th.Tensor:
        """ Returns a new view of the deque as a PyTorch tensor. """
        return self._buf[self._begin:self._end]
