from typing import Iterable, Tuple, Union, overload
from mltk.types import T, U

from ._internal import _NO_VALUE, _NoValue

__all__ = [
    "iter_nwise"
]

@overload
def iter_nwise(iterable: Iterable[T], n: int) -> Tuple[T, ...]: ...

@overload
def iter_nwise(iterable: Iterable[T], n: int, pad_value: U) -> Tuple[Union[T, U], ...]: ...

def iter_nwise(iterable: Iterable[T], n: int, pad_value: Union[U, _NoValue] = _NO_VALUE
    ) -> Tuple[Union[T, U], ...]:
    raise NotImplementedError
