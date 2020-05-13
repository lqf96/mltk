from typing import Any, Iterable, Mapping, Optional, Tuple, Union
from mltk.types import Shape

import torch as th

from .deque import Deque

__all__ = [
    "RecArray",
    "RecDeque",
    "RecDequeSchema"
]

class RecArray():
    def __init__(self, columns):
        object.__setattr__(self, "_columns", dict(columns))
    
    def __len__(self):
        return len(next(iter(self._columns.values())))

    def __getattr__(self, attr: str) -> Any:
        return self._columns[attr]

    def __setattr__(self, attr: str, values):
        self._columns[attr] = values

    def __getitem__(self, indices) -> Any:
        return self.map_columns(lambda column: column[indices])
    
    def __bool__(self) -> bool:
        return self.__len__()>0

    @classmethod
    def from_kwargs(cls, **columns) -> "RecArray":
        return cls(columns)

    def copy(self) -> "RecArray":
        return RecArray(self._columns)

    def map_columns(self, f) -> "RecArray":
        return RecArray((
            (col_name, f(col)) for col_name, col in self._columns.items()
        ))

    def to(self, *args, **kwargs) -> "RecArray":
        return self.map_columns(lambda column: column.to(*args, **kwargs))
    
    def unsqueeze(self, dim) -> "RecArray":
        return self.map_columns(lambda column: column.unsqueeze(dim))

# Schema of a record deque column
_ColumnSchema = Union[
    th.dtype,
    Tuple[Shape, th.dtype]
]
# Schema of a record deque
RecDequeSchema = Union[
    Mapping[str, _ColumnSchema],
    Iterable[Tuple[str, _ColumnSchema]]
]

def _col_from_schema(col_schema: _ColumnSchema, max_len: Optional[int]) -> Deque:
    # Column stores scalar values if shape is not provided
    if isinstance(col_schema, th.dtype):
        col_schema = ((), col_schema)
    
    return Deque(col_schema[0], col_schema[1], max_len=max_len)

class RecDeque(RecArray):
    @classmethod
    def from_schema(cls, schema: RecDequeSchema, max_len: Optional[int] = None) -> "RecDeque":
        # Iterate through the schema mapping
        if isinstance(schema, Mapping):
            schema = schema.items()
        
        return cls((
            (col_name, _col_from_schema(col_schema, max_len)) \
            for col_name, col_schema in schema
        ))

    def append(self, **kwargs: Any):
        for key, value in kwargs.items():
            self._columns[key].append(value)
