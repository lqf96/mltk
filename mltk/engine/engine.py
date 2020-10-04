from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, \
    List, Optional, Protocol, Set, Tuple, TypeVar, Union, overload
from mltk.types import Args, Kwargs, StrDict

import inspect, asyncio as aio
from dataclasses import dataclass

from mltk.util import create_bg_task
from .events import AbstractEvents, Events

if TYPE_CHECKING:
    from .metrics import Metric

__all__ = [
    "Engine",
    "EventFilter",
    "EventHandler",
    "TerminationCond",
    "State",
    "every",
    "run_n_epochs",
    "run_n_iters"
]

EventFilter = Callable[["State", AbstractEvents], bool]

EventHandler = Callable[..., Any]

TerminationCond = Callable[["State"], bool]

_HandlerT = TypeVar("_HandlerT", bound=EventHandler)

class _OnDecorator(Protocol[_HandlerT]):
    def __call__(self, f: _HandlerT) -> _HandlerT: ...

@dataclass(frozen=True)
class _HandlerInfo:
    """\
    Extra information for registered event handlers.

    Attributes:
        filter: Event filter function.
        gated_events: Events that are gated on the completion of this event handler.
            Only applies to asynchronous event handlers.
        args: Arguments of the handler function.
        kwargs: Keyword arguments of the handler function.
    """
    filter: Optional[EventFilter]
    gated_events: Tuple[AbstractEvents, ...]
    args: Tuple[Any, ...]
    kwargs: "StrDict"

class State():
    """\
    Execution state of the engine.

    Attributes:
        epochs: Number of total epochs (or episodes for RL).
        iterations: Number of total iterations (or steps for RL).
        epoch_iterations: Number of iterations within the current epoch.
        event_to_attr: Event to attribute name mapping.
        metrics: Values of named metrics registered with the engine.
    """
    def __init__(self, *, epochs: int, iterations: int, epoch_iterations: int):
        self.epochs = epochs
        self.iterations = iterations
        self.epoch_iterations = epoch_iterations
        
        self.event_to_attr: Dict[AbstractEvents, str] = {
            Events.ITER_COMPLETED: "iterations",
            Events.EPOCH_COMPLETED: "epochs"
        }
    
    def count_for(self, event: AbstractEvents) -> int:
        event_attr = self.event_to_attr[event]
        return getattr(self, event_attr)

class Engine():
    def __init__(self, main_func: Callable[..., Awaitable[Any]]):
        ## Engine state
        self.state: Optional[State] = None
        ## Main function of the engine
        self.main_func = main_func

        ## Backend task of the engine
        self._bg_tasks: Set["aio.Task[Any]"] = set()
        ## Event handlers mapping
        self._handlers: Dict[AbstractEvents, Dict[EventHandler, _HandlerInfo]] = {}
        ## Pending tasks events are gated on
        self._gate_tasks: Dict[AbstractEvents, List["aio.Task[Any]"]] = {}

        self._metric_groups: Dict[str, Set["Metric[Any]"]] = {}

    async def emit(self, event: AbstractEvents):
        state = self.state
        # Attempt to emit event when the engine is not running
        if state is None:
            raise RuntimeError("cannot emit event when the engine is not running")

        gate_tasks = self._gate_tasks.get(event, [])
        # Wait for the gate tasks of the event to complete
        if gate_tasks:
            await aio.wait(gate_tasks)

        # Get callback store for event
        handlers = self._handlers.get(event)
        if handlers:
            # Invoke all handlers
            for handler, info in handlers.items():
                filter = info.filter
                # Check event filter (if available)
                if filter and not filter(state, event):
                    continue
                
                # Invoke event handler
                result = handler(self, *info.args, **info.kwargs)
                # Handler is asynchronous
                if inspect.isawaitable(result):
                    task = create_bg_task(self._bg_tasks, result)

                    # Gate given events on the completion of this asynchronous event handler
                    for gated_event in info.gated_events:
                        gate_tasks = self._gate_tasks.setdefault(gated_event, [])
                        gate_tasks.append(task)

    @overload
    def on(self, event: AbstractEvents, handler: _HandlerT, *, oneshot: bool = False,
        filter: Optional[EventFilter] = None, gated_events: Iterable[AbstractEvents] = (),
        args: Args = (), kwargs: Kwargs = {}) -> _HandlerT: ...

    @overload
    def on(self, event: AbstractEvents, *, oneshot: bool = False,
        filter: Optional[EventFilter] = None, gated_events: Iterable[AbstractEvents] = (),
        args: Args = (), kwargs: Kwargs = {}) -> _OnDecorator[_HandlerT]: ...

    def on(self, event: AbstractEvents, handler: Optional[_HandlerT] = None, *,
        oneshot: bool = False, filter: Optional[EventFilter] = None,
        gated_events: Iterable[AbstractEvents] = (), args: Args = (), kwargs: Kwargs = {}
        ) -> Union[_HandlerT, _OnDecorator[_HandlerT]]:
        # Return a decorator that registers event handler when provided
        if handler is None:
            return lambda _handler: self.on(
                event,
                _handler,
                oneshot=oneshot,
                filter=filter,
                gated_events=gated_events,
                args=args,
                kwargs=kwargs
            )
        
        # Get callback store for event
        handlers = self._handlers.setdefault(event, {})
        # Store event handler and its callback information
        handlers[handler] = _HandlerInfo(
            filter=filter,
            gated_events=tuple(gated_events),
            args=tuple(args),
            kwargs=dict(kwargs)
        )

        # Event handler can be used to unregister itself from the engine
        return handler

    async def epochs(self) -> AsyncIterator[int]:
        termination_cond = self.termination_cond
        state = self.state

        # Attempt to create epochs iterator when the engine is not running
        if state is None:
            raise RuntimeError(
                "cannot create epochs iterator when the engine is not running"
            )

        while True:
            # End of previous epoch
            if state.epochs>0:
                await self.emit(Events.EPOCH_COMPLETED)
            # Check termination condition if it exists
            if termination_cond and termination_cond(state):
                return
            
            # Update number of epochs
            state.epochs += 1
            # Reset number of iterations in epoch
            state.epoch_iterations = 0

            # Emit epoch started event
            await self.emit(Events.EPOCH_STARTED)
            # Yield number of epochs
            yield state.epochs

    async def iterations(self) -> AsyncIterator[int]:
        termination_cond = self.termination_cond
        state = self.state

        # Attempt to create iterations iterator when the engine is not running
        if state is None:
            raise RuntimeError(
                "cannot create iterations iterator when the engine is not running"
            )

        while True:
            # End of previous iteration
            if state.epoch_iterations>0:
                await self.emit(Events.ITER_COMPLETED)
            # Check termination condition if it exists
            if termination_cond and termination_cond(state):
                return
            
            # Update number of iterations
            state.iterations += 1
            state.epoch_iterations += 1

            # Emit iteration started event
            await self.emit(Events.ITER_STARTED)
            # Yield number of iterations in epoch
            yield state.epoch_iterations

    async def run_async(self, *args: Any, termination_cond: Optional[TerminationCond] = None,
        state: Optional[State] = None, **kwargs: Any):
        ## Optional termination condition of the engine
        self.termination_cond = termination_cond
        ## State of the engine
        self.state = State(
            epochs=0,
            iterations=0,
            epoch_iterations=0
        ) if state is None else state
        ## Named metric instances registered with the engine
        self.metrics: Dict[str, "Metric[Any]"] = {}

        # Emit engine started event
        await self.emit(Events.STARTED)
        # Run engine's main function
        await self.main_func(self, *args, **kwargs)
        # Emit engine completed event
        await self.emit(Events.COMPLETED)
    
    def run(self, *args: Any, loop: Optional[aio.AbstractEventLoop] = None,
        **kwargs: Any):
        # Use the default event loop if no event loop is provided
        if loop is None:
            loop = aio.get_event_loop()
        # Run engine inside the event loop
        loop.run_until_complete(self.run_async(*args, **kwargs))

def every(interval: int) -> EventFilter:
    def every_filter(state: State, event: AbstractEvents) -> bool:
        return state.count_for(event)%interval==0
    
    return every_filter

def run_n_epochs(n_epochs: int) -> TerminationCond:
    return lambda state: state.epochs>n_epochs

def run_n_iters(n_iterations: int) -> TerminationCond:
    return lambda state: state.iterations>n_iterations
