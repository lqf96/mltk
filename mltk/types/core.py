from typing import Any, Callable, Dict, Iterable, Mapping, TypeVar

__all__ = [
    "T",
    "U",
    "Args",
    "Kwargs",
    "StrDict",
    "Factory"
]

## Generic type parameter
T = TypeVar("T")
U = TypeVar("U")

Args = Iterable[Any]
Kwargs = Mapping[str, Any]
StrDict = Dict[str, Any]

Factory = Callable[..., T]