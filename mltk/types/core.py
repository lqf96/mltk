from typing import Any, Dict, Iterable, Mapping, Tuple, TypeVar, Union

__all__ = [
    "T",
    "U",
    "Args",
    "IntoMapping",
    "Kwargs",
    "Pair",
    "StrDict",
]

# Generic type parameter
T = TypeVar("T")
U = TypeVar("U")

Args = Iterable[Any]
Kwargs = Mapping[str, Any]
StrDict = Dict[str, Any]

Pair = Tuple[T, T]

IntoMapping = Union[
    Iterable[Tuple[T, U]],
    Mapping[T, U]
]