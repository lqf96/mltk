from typing import TypeVar, Iterable, Mapping, Dict, Any

__all__ = [
    "T",
    "Args",
    "Kwargs",
    "StrDict"
]

## Generic type parameter
T = TypeVar("T")

Args = Iterable[Any]
Kwargs = Mapping[str, Any]
StrDict = Dict[str, Any]