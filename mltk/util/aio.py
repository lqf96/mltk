from __future__ import annotations

from typing import Callable, Optional, Any
from collections.abc import Awaitable
from mltk.types import T

import inspect
import asyncio as aio
from asyncio import AbstractEventLoop, Future, Task

__all__ = [
    "create_bg_task",
    "set_immediate",
    "resolved",
    "rejected"
]

_ErrorHandler = Callable[[BaseException], None]

def create_bg_task(task_set: set[Task[Any]], awaitable: Awaitable[Any], *,
    error_handler: Optional[_ErrorHandler] = None, loop: Optional[AbstractEventLoop] = None
    ) -> Task[Any]:
    """\
    Creates and returns a background task from an awaitable object.

    The task is stored in the provided task set to avoid being garbage collected.
    It is automatically removed from the task set when fulfilled.

    Args:
        task_set: Task set to store background tasks.
        awaitable: Awaitable object to create background task from.
        loop: The AsyncIO event loop to use.
    
    Returns:
        The background task object.
    """
    loop = loop or aio.get_event_loop()

    def _on_completed(task: Task[Any]):
        error = task.exception()
        # Task is rejected with error
        if error:
            # Handle task error with given handler
            if error_handler:
                error_handler(error)
            # Rethrow error if no handler is given
            else:
                raise error

        # Remove task from set
        task_set.remove(task)

    # Create task from the awaitable
    task = loop.create_task(awaitable) if inspect.iscoroutine(awaitable) else awaitable
    # Store the task in the task set (to avoid being garbage collected)
    task_set.add(task)
    # Perform clean-up on completion of the task
    task.add_done_callback(_on_completed)

    return task

def set_immediate(loop: Optional[AbstractEventLoop] = None) -> Future[None]:
    """\
    Create a future that will be resolved on the next tick of the event loop.

    Args:
        loop: The AsyncIO event loop to use.
    
    Returns:
        A future that will be resolved on the next tick of the event loop.
    """
    loop = loop or aio.get_event_loop()
    
    future: aio.Future[None] = loop.create_future()
    # Use "call_soon" to provide the "next tick" semantics
    loop.call_soon(future.set_result, None)

    return future

async def resolved(value: T) -> T:
    """\
    Create a coroutine that is resolved with given value.

    Returns:
        A coroutine resolved with given value.
    """
    return value

async def rejected(error: BaseException) -> Any:
    """\
    Create a coroutine that is rejected with given error.

    Returns:
        A coroutine rejected with given error.
    """
    raise error
