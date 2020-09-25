from typing import Any, Callable, Dict, Iterable, Mapping, Tuple, TypeVar, Union

__all__ = [
    "T",
    "U",
    "Args",
    "IntoMapping",
    "Kwargs",
    "StrDict",
    "Factory"
]

# Generic type parameter
T = TypeVar("T")
U = TypeVar("U")

Args = Iterable[Any]
Kwargs = Mapping[str, Any]
StrDict = Dict[str, Any]

Factory = Callable[..., T]

IntoMapping = Union[
    Iterable[Tuple[T, U]],
    Mapping[T, U]
]