from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional, Union, overload
from mltk.types import Args, Kwargs
from collections.abc import Awaitable, Callable, Iterable

import asyncio as aio, inspect
from asyncio import AbstractEventLoop, Task

import mltk.util as mu
from .types import AbstractEvent, AbstractEventKind, Events, EventFilter, EventHandler, State, \
    TerminationCond, _HandlerInfo, _HandlerT, _OnDecorator, _StateT

if TYPE_CHECKING:
    from .metrics import Metric

__all__ = [
    "Engine",
    "EventFilter",
    "EventHandler",
    "TerminationCond",
    "State",
    "run_n_epochs",
    "run_n_iters"
]

class Engine(Generic[_StateT]):
    def __init__(self, main_func: Callable[..., Awaitable[Any]],
        state_factory: Callable[..., _StateT] = State):
        self.main_func = main_func
        self.state_factory = state_factory
        self.term_cond: Optional[TerminationCond] = None
        self.state: Optional[_StateT] = None

        # Active background tasks
        self._bg_tasks: set[Task[Any]] = set()
        # Event handlers mapping
        self._handlers: dict[AbstractEventKind, dict[EventHandler, _HandlerInfo[_StateT]]] = {}
        # Mapping of tasks that an event is gated on
        self._gated_on_tasks: dict[AbstractEventKind, list[Task[Any]]] = {}

        self._metric_groups: dict[str, set["Metric[Any]"]] = {}

    def _handle_error(self, error: BaseException):
        handle_error_task = self.emit(Events.EXCEPTION_RAISED, error)
        # Handle engine error in the background
        # (No nested error handler is provided to avoid recursion errors)
        mu.create_bg_task(self._bg_tasks, handle_error_task)

    def ensure_state(self) -> _StateT:
        state = self.state
        # Check that state is not empty
        if state is None:
            raise RuntimeError("operation not permitted unless engine is running")
        
        return state

    async def start_iteration(self) -> int:
        state = self.ensure_state()

        # Update number of epoch and total iterations
        state.n_iters += 1
        state.n_epoch_iters += 1
        # Emit iteration started event
        await self.emit(Events.ITER_STARTED)

        return state.n_epoch_iters

    async def end_iteration(self) -> bool:
        state = self.ensure_state()
        term_cond = self.term_cond

        # Emit iteration completed event
        await self.emit(Events.ITER_COMPLETED)
        # Check termination condition
        return term_cond(state) if term_cond else False

    async def start_epoch(self) -> int:
        state = self.ensure_state()

        # Beginning of a new episode
        if state.n_epoch_iters==0:
            # Update number of epochs
            state.n_epochs += 1
            # Emit epoch started event
            await self.emit(Events.EPOCH_STARTED)
        
        return state.n_epochs
    
    async def end_epoch(self):
        state = self.ensure_state()

        # Emit epoch completed event
        await self.emit(Events.EPOCH_COMPLETED)
        # Reset number of epoch iterations
        state.n_epoch_iters = 0

    async def emit(self, event_kind: AbstractEventKind, *args: Any, **kwargs: Any):
        state = self.ensure_state()

        gated_on_tasks = self._gated_on_tasks.pop(event_kind, None)
        # Wait for the gate tasks of the event to complete
        if gated_on_tasks:
            await aio.wait(gated_on_tasks)

        # Get callbacks for event
        handlers = self._handlers.get(event_kind)
        if handlers:
            for handler, info in handlers.items():
                filter = info.filter
                # Check event filter (if available)
                if filter and not filter(state, event_kind):
                    continue
                
                # Invoke event handler
                result = handler(self, *args, *info.args, **kwargs, **info.kwargs)
                # Handler is asynchronous
                if inspect.isawaitable(result):
                    task = mu.create_bg_task(
                        self._bg_tasks, result, error_handler=self._handle_error
                    )

                    # Gate given events on the completion of this asynchronous event handler
                    for gate_event in info.gate_events:
                        gated_on_tasks = self._gated_on_tasks.setdefault(gate_event, [])
                        gated_on_tasks.append(task)

    @overload
    def on(self, event: AbstractEvent[_StateT], handler: _HandlerT, *, oneshot: bool = False,
        gate_events: Iterable[AbstractEventKind] = (), args: Args = (), kwargs: Kwargs = {}
        ) -> _HandlerT: ...

    @overload
    def on(self, event: AbstractEvent[_StateT], *, oneshot: bool = False,
        gate_events: Iterable[AbstractEventKind] = (), args: Args = (), kwargs: Kwargs = {}
        ) -> _OnDecorator[_HandlerT]: ...

    def on(self, event: AbstractEvent[_StateT], handler: Optional[_HandlerT] = None, *,
        oneshot: bool = False, gate_events: Iterable[AbstractEventKind] = (), args: Args = (),
        kwargs: Kwargs = {}) -> Union[_HandlerT, _OnDecorator[_HandlerT]]:
        # Return a decorator that registers event handler when provided
        if handler is None:
            return lambda _handler: self.on(
                event,
                _handler,
                oneshot=oneshot,
                gate_events=gate_events,
                args=args,
                kwargs=kwargs
            )
        
        # Get callbacks store for event
        handlers = self._handlers.setdefault(event.kind, {})
        # Store event handler and related callback information
        handlers[handler] = _HandlerInfo(
            filter=event.filter,
            gate_events=tuple(gate_events),
            args=tuple(args),
            kwargs=dict(kwargs)
        )

        # Event handler can be used to unregister itself from the engine
        return handler

    async def run_async(self, *args: Any, term_cond: Optional[TerminationCond] = None,
        state: Optional[_StateT] = None, **kwargs: Any) -> Any:
        self.term_cond = term_cond
        self.state = state or self.state_factory(n_epochs=0, n_iters=0, n_epoch_iters=0)

        # Emit engine started event
        await self.emit(Events.STARTED)
        # Run engine's main function
        result = await self.main_func(self, *args, **kwargs)
        # Emit engine completed event
        await self.emit(Events.COMPLETED)

        return result
    
    def run(self, *args: Any, loop: Optional[AbstractEventLoop] = None,
        **kwargs: Any) -> Any:
        # Defaults to AsyncIO's default event loop
        loop = loop or aio.get_event_loop()
        # Run engine inside the event loop
        return loop.run_until_complete(self.run_async(*args, **kwargs))

def run_n_epochs(n_epochs: int) -> TerminationCond:
    return lambda state: state.n_epochs>n_epochs

def run_n_iters(n_iters: int) -> TerminationCond:
    return lambda state: state.n_iters>n_iters
