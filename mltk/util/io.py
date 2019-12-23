from typing import Awaitable, Generator, Any
from mltk.typing import T

import asyncio, functools
from asyncio import AbstractEventLoop, Future

__all__ = [
    "spawn_task",
    "set_immediate",
    "delay",
    "resolved",
    "rejected"
]

def spawn_task(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        asyncio.create_task(func(*args, **kwargs))
    return wrapper

def set_immediate(loop: Optional[AbstractEventLoop] = None) -> "Future[None]":
    if loop is None:
        loop = asyncio.get_event_loop()
    
    f = loop.create_future()
    loop.call_soon(f.set_result, None)
    return f

def delay(interval: float, loop: Optional[AbstractEventLoop] = None) -> "Future[None]":
    if loop is None:
        loop = asyncio.get_event_loop()

    f = loop.create_future()
    loop.call_later(interval, f.set_result, None)
    return f

async def resolved(value: T) -> T:
    """ Return a coroutine that yields given value. """
    return value

async def rejected(error: Exception) -> Any:
    """ Return a coroutine that raises given exception. """
    raise error

