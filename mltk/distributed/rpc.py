from typing import Generic, Type, Dict, Set, Callable, Any, Awaitable, Union, Optional
from mltk.typing import T, Args, Kwargs, StrDict

import inspect, asyncio, time
from asyncio import Future

from torch.distributed.rpc import RRef, WorkerInfo, get_worker_info, rpc_async, init_rpc, \
    shutdown

import mltk.util as mu
from mltk.util.helpers import _NO_VALUE

__all__ = [
    "RemoteValue",
    "RemoteError",
    "RPCResult",
    "rpc",
    "run_rpc"
]

class RemoteValue(Generic[T]):
    method_prefix = "m_"
    magic_prefix = "s_"

    def __init__(self, value: T, method_prefix: Optional[str] = None,
        magic_prefix: Optional[str] = None):
        self._value = value._value if isinstance(value, RemoteValue) else value

        if method_prefix!=None:
            self.method_prefix = method_prefix
        if magic_prefix!=None:
            self.magic_prefix = magic_prefix

    @staticmethod
    def _pass_string_by_val(s: str) -> bool:
        """ Check whether given string should be passed by value. """
        # Pass string with <= 50 characters by value
        return len(s)<=50

    @staticmethod
    def _pass_bytes_by_val(b: bytes) -> bool:
        """ Check whether given bytes should be passed by value. """
        # Pass data <= 200 bytes by value
        return len(b)<200

    @classmethod
    def _pass_tuple_by_val(cls, t: tuple) -> bool:
        """ Check whether given tuple should be passed by value. """
        # Pass tuple with > 5 elemens by reference
        if len(t)>5:
            return False
        # All elements must be preferred to be passed by value
        for element in t:
            if not cls._pass_by_val(element):
                return False
        return True

    @classmethod
    def _pass_by_val(cls, obj: Any) -> bool:
        """ Check whether given object should be passed by value. """
        obj_type = type(obj)
        # Not specified to be passed by value
        if obj_type not in cls._BY_VAL_TYPES:
            return False

        by_val_cond = cls._BY_VAL_COND.get(obj_type)
        # Check pass-by-value condition if it exists
        return by_val_cond(obj) if by_val_cond else True

    def _call_remote(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """ Remote wrapper function for calling member method on wrapped value. """
        bound_method = getattr(self.value, method_name)
        return bound_method(*args, **kwargs)

    def _get_value_remote(self):
        return self.value

    def __getstate__(self) -> StrDict:
        value = self._value
        # Lazily create reference for value during serialization
        if not isinstance(value, RRef) and not self._pass_by_val(value):
            self._value = RRef(value)
        return self.__dict__

    def __getitem__(self, key: Any) -> Awaitable[Any]:
        return self.getitem(key)

    def __getattr__(self, attr: str) -> Union[Callable[..., Awaitable[Any]], Awaitable[Any]]:
        # Member method call
        method_prefix = self.method_prefix
        if attr.startswith(method_prefix):
            method_name = attr[len(method_prefix):]
            return lambda *args, **kwargs: self.call(method_name, args=args, kwargs=kwargs)

        # Magic method call
        magic_prefix = self.magic_prefix
        if attr.startswith(magic_prefix):
            method_name = f"__{attr[len(magic_prefix):]}__"
            return lambda *args, **kwargs: self.call(method_name, args=args, kwargs=kwargs)

        # Attribute look-up
        return self.getattr(attr)

    @property
    def is_owner(self) -> bool:
        value = self._value
        return value.is_owner() if isinstance(value, RRef) else True

    @property
    def value(self) -> T:
        value = self._value
        return value.local_value() if isinstance(value, RRef) else value

    @property
    def owner(self) -> WorkerInfo:
        value = self._value
        return value.owner() if isinstance(value, RRef) else get_worker_info()

    def get_value(self) -> Awaitable[T]:
        if self.is_owner:
            return mu.resolved(self.value)
        else:
            return rpc(
                self._value.owner, mu.identity, args=(self,), result_as_ref=False
            )

    def call(self, method_name: str, args: Args, kwargs: Kwargs, params_by_ref: bool = False,
        result_as_ref: bool = True) -> "RPCResult[T]":
        return rpc(
            self.owner, RemoteValue._call_remote,
            args=(self, method_name, *args),
            kwargs=kwargs,
            params_by_ref=params_by_ref,
            result_as_ref=result_as_ref
        )

    async def getitem(self, key: Any, default: Any = _NO_VALUE, as_ref: bool = True) -> Any:
        """
        Get item from the remote object.

        :param key: Key of the item.
        :param default: Fallback value if given item is not found.
        :param as_ref: Whether to return result as reference or not.
        :raises KeyError: If item with given key does not exist.
        :raises AttributeError: If the remote object is not subscriptable.
        """
        try:
            return await self.call("__getitem__", args=(key,), result_as_ref=as_ref)
        # Item not found
        except KeyError as e:
            # No default value has been set
            if default is self._NO_VALUE:
                raise e
            else:
                return default

    def setitem(self, key: Any, value: Any) -> Awaitable[None]:
        """
        Set item on the remote object to given value.

        :param key: Key of the item.
        :param value: Value to be set.
        :raises AttributeError: If the remote object does not support item assignment.
        """
        return self.call("__setitem__", args=(key, value))

    def delitem(self, key: Any) -> Awaitable[None]:
        """
        Delete item of the remote object.

        :param key: Key of the item.
        :raises KeyError: If item with given key does not exist.
        :raises AttributeError: If the remote object does not support item deletion.
        """
        return self.call("__delitem__", args=(key,))

    async def getattr(self, attr: str, default: Any = _NO_VALUE, as_ref: bool = True) -> Any:
        """
        Get attribute from the remote object.

        :param attr: Name of the attribute.
        :param default: Fallback value if given attribute is not found.
        :param as_ref: Whether to return result as reference or not.
        :raises AttributeError: If given attribute is not found on object.
        """
        value = await rpc(
            self.owner, getattr, args=(self, attr, default), result_as_ref=as_ref
        )
        # Attribute not found and no default value has been set
        if value is _NO_VALUE:
            raise AttributeError(f"remote object has no attribute {attr}")
        else:
            return value

    def setattr(self, attr: str, value: Any) -> Awaitable[None]:
        """
        Set attribute on the remote object to given value.

        :param attr: Name of the attribute.
        :param value: Value to be set.
        :raises AttributeError: If the remote object is immutable or frozen.
        """
        return rpc(self.owner, setattr, args=(self, attr, value))

    def delattr(self, attr: str) -> Awaitable[None]:
        """
        Delete attribute of the remote object.

        :param attr: Name of the attribute.
        :raises AttributeError: If the remote object is immutable or frozen.
        """
        return rpc(self.owner, delattr, args=(self, attr))

    _BY_VAL_TYPES: Set[type] = set((
        # Prelude built-in types
        int,
        float,
        complex,
        bool,
        str,
        bytes,
        slice,
        range,
        tuple,
        # Other built-in types
        type(None),
        type(...)
    ))

    _BY_VAL_COND = {
        str: _pass_string_by_val,
        bytes: _pass_bytes_by_val,
        tuple: _pass_tuple_by_val
    }

class RemoteError(Exception, RemoteValue):
    def __init__(self, error: Exception):
        # Initialize base classes
        Exception.__init__(*error.args)
        RemoteValue.__init__(error)

class RPCResult(Awaitable[T]):
    def __init__(self, inner: Awaitable[T]):
        self._inner = inner

    def __await__(self):
        return self._inner.__await__()

async def _exec_rpc(func, args: Args, kwargs: Kwargs, as_ref: bool) -> Any:
    """ Execute RPC and await on the result if it is awaitable. """
    try:
        result = func(*args, **kwargs)
        # Result is awaitable
        if inspect.isawaitable(result):
            result = await result
        # Pass result by reference
        if as_ref:
            result = RemoteValue(result)
        return result

    except Exception as error:
        # Pass error by reference
        if as_ref:
            error = RemoteError(error)
        raise error

@mu.spawn_task
async def _exec_rpc_remote(func, args: Args, kwargs: Kwargs, as_ref: bool,
    future_ref):
    """ Remote wrapper function for an asynchronous RPC. """
    try:
        result = await _exec_rpc(func, args, kwargs, as_ref)
        # Inform caller of RPC result
        rpc_async(future_ref.owner(), Future.set_result, args=(future_ref, result))

    except Exception as error:
        # Inform caller of RPC error
        rpc_async(future_ref.owner(), Future.set_exception, args=(future_ref, error))

def rpc(target: Union[str, WorkerInfo], func, args: Args = (), kwargs: Kwargs = {},
    params_by_ref: bool = True, result_as_ref: bool = True) -> RPCResult[Any]:
    # Pass parameters by reference
    if params_by_ref:
        args = [RemoteValue(arg) for arg in args]
        kwargs = {key: RemoteValue(value) for key, value in kwargs.items()}

    # Get current and target worker
    target_worker = get_worker_info(target)
    current_worker = get_worker_info()
    # RPC target is current node
    if target_worker==current_worker:
        return RPCResult(_exec_rpc(func, args, kwargs, result_as_ref))

    result_future = Future()
    # Perform RPC on remote node
    rpc_async(
        target_worker, _exec_rpc_remote,
        args=(func, args, kwargs, result_as_ref, RRef(result_future))
    )
    return RPCResult(result_future)

async def _yield_gil():
    """ Yield GIL to other threads when this coroutine is active. """
    while True:
        # Yield GIL to other threads
        time.sleep(0)
        # Yield to other tasks
        await mu.set_immediate()

def run_rpc(ready_callback, **kwargs):
    # Initialize PyTorch RPC
    init_rpc(**kwargs)

    # Create AsyncIO event loop
    loop = asyncio.new_event_loop()
    loop.create_task(_yield_gil())
    loop.run_until_complete(ready_callback())

    # Shutdown PyTorch RPC
    shutdown()
