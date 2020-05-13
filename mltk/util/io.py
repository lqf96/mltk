from typing import Optional, Awaitable, Any, Set
from mltk.types import T

import asyncio as aio
from functools import partial
from asyncio import AbstractEventLoop, Future, Task

__all__ = [
    "create_bg_task",
    "set_immediate",
    "resolved",
    "rejected"
]

def _remove_bg_task(task_set: Set["Task[T]"], task: "Task[T]"):
    error = task.exception()
    # Task is rejected with error
    if error is not None:
        raise error
    
    # Remove task from task set
    task_set.remove(task)

def create_bg_task(task_set: Set["Task[T]"], awaitable: Awaitable[T], *,
    loop: Optional[AbstractEventLoop] = None) -> "Task[T]":
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
    if loop is None:
        loop = aio.get_event_loop()

    # Create task from the awaitable
    task = loop.create_task(awaitable)
    # Store the task in the task set (to avoid being garbage collected)
    task_set.add(task)
    # Remove the task from the task set when it is done
    task.add_done_callback(partial(_remove_bg_task, task_set=task_set))

    return task

def set_immediate(loop: Optional[AbstractEventLoop] = None) -> "Future[None]":
    """\
    Create a future that will be resolved on the next tick of the event loop.

    Args:
        loop: The AsyncIO event loop to use.
    
    Returns:
        A future that will be resolved on the next tick of the event loop.
    """
    if loop is None:
        loop = aio.get_event_loop()
    
    future = loop.create_future()
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

async def rejected(error: Exception) -> Any:
    """\
    Create a coroutine that is rejected with given error.

    Returns:
        A coroutine rejected with given error.
    """
    raise error
