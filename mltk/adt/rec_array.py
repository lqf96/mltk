from typing import Any, Iterator, Mapping, Optional, Tuple, Union

from numpy.core.records import array
from numpy.lib.shape_base import column_stack
from mltk.types import Device, IntoMapping, Shape

from types import MappingProxyType

import torch as th

import mltk.util as mu
from .deque import Deque

__all__ = [
    "RecArray",
    "RecDeque",
    "RecDequeSchema"
]

class RecArray():
    _array_meta: th.Tensor

    __slots__ = ("_array_meta", "__dict__")

    def __init__(self, data: IntoMapping[str, Any] = {}, shape: Optional[Shape] = None,
        batch_dims: int = 0, device: Device = "cpu", _array_meta: Optional[th.Tensor] = None,
        _skip_validation: bool = False):
        # Add columns to the record array
        self.__dict__.update(data)

        if _array_meta is None:
            # Create meta tensor from inferred batch shape
            batch_shape = self._infer_batch_shape(batch_dims) \
                if shape is None \
                else mu.as_size(shape)
            _array_meta = th.empty((0, *batch_shape), device=mu.as_device(device))
        # Bypass column checks when storing meta tensor
        self._set_array_meta(_array_meta)

        if not _skip_validation:
            # Validate and convert column data against meta tensor
            for col_name, col_data in self.columns.items():
                setattr(self, col_name, col_data)

    def _set_array_meta(self, array_meta: th.Tensor):
        self.__class__._array_meta.__set__(self, array_meta)

    def _infer_batch_shape(self, batch_dims: int) -> th.Size:
        try:
            col_name, col_data = next(iter(self.columns.items()))
            # Column does not have enough dimensions
            if col_data.ndim<batch_dims:
                raise ValueError(
                    f"column '{col_name}' does not have enough dimensions "
                    f"(expect at least {batch_dims}, got {col_data.ndim})"
                )
            
            return col_data.shape[:batch_dims]
        except StopIteration:
            return th.Size()

    # Record array is not hashable
    __hash__ = None

    def __len__(self) -> int:
        if self.ndim==0:
            raise TypeError("length unavailable for 0-d record array")
        else:
            return self._array_meta.shape[1]

    def __setattr__(self, col_name: str, col_data):
        array_meta = self._array_meta

        batch_shape = array_meta.shape[1:]
        col_batch_shape = col_data.shape[:len(batch_shape)]
        # Column has conflicting batch shape
        if col_batch_shape!=batch_shape:
            raise ValueError(
                f"conflicting batch shape of column '{col_name}' "
                f"(expect {batch_shape}, got {col_batch_shape})"
            )

        # Convert column to current device and then store it
        self.__dict__[col_name] = col_data.to(device=array_meta.device)

    def __getitem__(self, indices: Any) -> "RecArray":
        # Insert a slice-all index for the first dummy dimension
        meta_indices = (mu.SLICE_ALL, *indices) \
            if isinstance(indices, tuple) \
            else (mu.SLICE_ALL, indices)
        
        # Check indices validity by doing a fictional slice on the meta tensor,
        # then perform slicing on all columns
        return self.map_data(
            lambda col_data: col_data[indices],
            lambda array_meta: array_meta[meta_indices],
            _skip_validation=True
        )

    def __iter__(self) -> Iterator["RecArray"]:
        return (self[i] for i in range(len(self)))

    @property
    def device(self) -> th.device:
        return self._array_meta.device

    @property
    def ndim(self) -> int:
        return self._array_meta.dim()-1

    @property
    def shape(self) -> th.Size:
        return self._array_meta.shape[1:]

    @property
    def columns(self) -> Mapping[str, Any]:
        return MappingProxyType(self.__dict__)

    def map_data(self, func, func_meta=None, _skip_validation: bool = False):
        array_meta = self._array_meta
        
        # Compute the meta tensor for new record array
        func_meta = func_meta or func
        new_array_meta = func_meta(array_meta)

        # Create new record array with transformed column data
        return RecArray(
            ((col_name, func(col_data)) for col_name, col_data in self.columns.items()),
            _array_meta=new_array_meta,
            _skip_validation=_skip_validation
        )

    def to(self, device) -> "RecArray":
        return self.map_data(
            lambda col_data: col_data.to(device=device),
            _skip_validation=True
        )

# Schema of a record deque column
_ColumnSchema = Union[
    th.dtype,
    Tuple[Shape, th.dtype],
    "RecDequeSchema"
]
# Schema of a record deque
RecDequeSchema = IntoMapping[str, _ColumnSchema]

class RecDeque(RecArray):
    _EMPTY_CACHE = th.empty((0,))

    @classmethod
    def _col_from_schema(cls, col_schema: _ColumnSchema, device: Device,
        max_len: Optional[int]):
        # Column stores scalar values if shape is not provided
        if isinstance(col_schema, th.dtype):
            return Deque((), dtype=col_schema, device=device, max_len=max_len)
        elif isinstance(col_schema, tuple):
            col_shape, col_dtype = col_schema
            return Deque(col_shape, col_dtype, device, max_len=max_len)
        else:
            return cls.from_schema(col_schema, device, max_len=max_len)

    @classmethod
    def from_schema(cls, schema: RecDequeSchema, device: Device = "cpu",
        max_len: Optional[int] = None) -> "RecDeque":
        if isinstance(schema, Mapping):
            schema = schema.items()
        
        return cls(
            (
                (col_name, cls._col_from_schema(col_schema, device, max_len)) \
                for col_name, col_schema in schema
            ),
            _array_meta=th.empty((0, 0), device=th.device(device)),
            _skip_validation=True
        )
    
    @property
    def _array_meta(self) -> th.Tensor:
        # Get cached array meta tensor first
        array_meta = super()._array_meta
        if array_meta is self._EMPTY_CACHE:
            # Fallback to synthesized meta tensor if cache is unavailable
            column = next(iter(self.columns.values()))
            array_meta = th.empty((0, len(column)), device=column.device)
            # Update cache
            self._set_array_meta(array_meta)
        
        return array_meta
    
    _array_meta = _array_meta.setter(RecArray._array_meta.__set__)

    def append(self, data: Optional[IntoMapping[str, Any]] = None, **kwargs: Any):
        if data is None:
            data = kwargs
        else:
            data = dict(data)
            data.update(kwargs)

        columns = self.columns
        # Column names mismatch
        if data.keys()!=columns.keys():
            raise ValueError("")
        # Append values for each column
        for col_name, value in data.items():
            columns[col_name].append(value)

        # Invalidate array meta cache
        self._set_array_meta(self._EMPTY_CACHE)
