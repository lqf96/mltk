from __future__ import annotations

from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, overload
from abc import abstractmethod
from enum import Enum

from dataclasses import dataclass, field

__all__ = [
    "AbstractEvent",
    "AbstractEventKind",
    "Event",
    "Events",
    "EventFilter",
    "EventHandler",
    "EventKind",
    "Every",
    "Once",
    "State",
    "TerminationCond"
]

# Generic engine state type
_StateT = TypeVar("_StateT", bound="State")
_StateT_contra = TypeVar("_StateT_contra", bound="State", contravariant=True)

class EventFilter(Protocol[_StateT_contra]):
    def __call__(self, state: _StateT_contra, kind: AbstractEventKind) -> bool: ...

@dataclass(frozen=True)
class Every(EventFilter):
    interval: int

    def __call__(self, state: State, kind: AbstractEventKind):
        return True if self.interval==1 \
            else state.count_for(kind)%self.interval==0

@dataclass(frozen=True)
class Once(EventFilter):
    count: int

    def __call__(self, state: State, kind: AbstractEventKind):
        return state.count_for(kind)==self.count

class AbstractEvent(Protocol[_StateT_contra]):
    @property
    def kind(self) -> AbstractEventKind: ...

    @property
    def filter(self) -> Optional[EventFilter[_StateT_contra]]: ...

class Event(AbstractEvent[_StateT]):
    __all__ = ("_kind", "_filter")

    def __init__(self, kind: AbstractEventKind, filter: Optional[EventFilter[_StateT]] = None):
        self._kind = kind
        self._filter = filter
    
    def __repr__(self):
        repr_str = repr(self._kind)
        if self._filter:
            repr_str += f"(filter={repr(self._filter)})"
        
        return repr_str

    def __eq__(self, other: AbstractEvent) -> bool:
        return self._kind==other.kind and self._filter==other.filter

    def __ne__(self, other: AbstractEvent) -> bool:
        return self._kind!=other.kind or self._filter!=other.filter

    def __hash__(self) -> int:
        return hash(self._kind)^hash(self._filter)

    @property
    def kind(self) -> AbstractEventKind:
        return self._kind

    @property
    def filter(self) -> Optional[EventFilter[_StateT]]:
        return self._filter

class AbstractEventKind(Enum):
    # `AbstractEventKind` implements `AbstractEvent`
    @property
    def kind(self) -> AbstractEventKind:
        return self

    @property
    def filter(self) -> None:
        return None
    
    @overload
    def __call__(self, *, every: int = 1) -> Event: ...

    @overload
    def __call__(self, *, once: Optional[int] = None) -> Event: ...

    def __call__(self, *, every: int = 1, once: Optional[int] = None,
        filter: Optional[EventFilter[_StateT]] = None) -> Event[_StateT]:
        if filter is None:
            # Apply every filter
            if every>1:
                filter = Every(interval=every)
            # Apply once filter
            elif once is not None:
                filter = Once(count=once)
        
        return Event(kind=self, filter=filter)

class EventKind(AbstractEventKind):
    STARTED = "started"
    COMPLETED = "completed"
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    ITER_STARTED = "iteration_started"
    ITER_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"

# Type alias for basic event kinds
Events = EventKind

# Event handler type
EventHandler = Callable[..., Any]
# Generic event handler type
_HandlerT = TypeVar("_HandlerT", bound=EventHandler)

_EVENT_ATTR_DEFAULT: dict[AbstractEventKind, str] = {
    Events.EPOCH_STARTED: "n_epochs",
    Events.EPOCH_COMPLETED: "n_epochs",
    Events.ITER_STARTED: "n_iters",
    Events.ITER_COMPLETED: "n_iters"
}

@dataclass
class State:
    n_epochs: int
    n_iters: int
    n_epoch_iters: int
    
    epoch_completed: bool = field(default=True, init=False)
    event_attr_map: dict[AbstractEventKind, str] = field(
        default_factory=_EVENT_ATTR_DEFAULT.copy, init=False
    )

    def count_for(self, kind: AbstractEventKind) -> int:
        attr = self.event_attr_map[kind]
        return getattr(self, attr)

@dataclass(frozen=True)
class _HandlerInfo(Generic[_StateT]):
    """\
    Extra information for registered event handlers.

    Attributes:
        filter: Event filter function.
        gate_events: Events that are gated on the completion of this event handler.
            Only applies to asynchronous event handlers.
        args: Arguments of the handler function.
        kwargs: Keyword arguments of the handler function.
    """
    filter: Optional[EventFilter[_StateT]]
    gate_events: tuple[AbstractEventKind, ...]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

class _OnDecorator(Protocol[_HandlerT]):
    def __call__(self, f: _HandlerT) -> _HandlerT: ...

# A function that determines whether the engine should be terminated or not
TerminationCond = Callable[[State], bool]
