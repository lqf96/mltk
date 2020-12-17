from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Union
from collections.abc import Callable, Iterable
from mltk.types import T, TensorT, Args, Kwargs, Numerical

from abc import ABC, abstractmethod
from itertools import chain

from ..engine import Engine
from ..types import AbstractEventKind, Events, _StateT

__all__ = [
    "Triggers",
    "Metric",
    "StateMetric",
    "LambdaMetric",
    "attach_dependencies"
]

Triggers = Dict[str, AbstractEventKind]

class Metric(ABC, Generic[T]):
    __slots__ = ("triggers", "_engine", "_name")

    def __init__(self, triggers: Triggers = {}):
        super().__init__()

        # Trigger points of the metric
        self.triggers = triggers
        # Engine the metric is attached to
        self._engine: Optional[Engine] = None
        # Name of the attached metric
        self._name: Optional[str] = None

    def _on_start(self, engine: Engine, groups: Iterable[str]):
        metric_groups = engine._metric_groups
        
        # Store metric instance to given metric groups
        for group_name in groups:
            metric_group = metric_groups.setdefault(group_name, set())
            metric_group.add(self)

    def _on_reset(self, _: Engine):
        self.reset()

    def _on_update(self, _: Engine):
        self.update()

    @property
    def attached(self) -> bool:
        """ Whether the metric is attached to an engine or not. """
        return self._engine is not None

    def reset(self) -> None:
        pass

    def update(self) -> None:
        pass

    @abstractmethod
    def compute(self) -> T:
        return NotImplementedError

    def attach(self, engine: Engine, name: str, groups: Union[None, str, Iterable[str]] = None):
        if self.attached:
            # Attempt to attach metric to a second engine
            if engine!=self._engine:
                raise ValueError("cannot attach metric to a different engine")
            else:
                return
        
        self._engine = engine
        self._name = name

        triggers = self.triggers
        # Attach (optional) reset event handler
        reset_event = triggers.get("reset")
        if reset_event is not None:
            engine.on(reset_event, self._on_reset)
        # Attach (optional) update event handler
        update_event = triggers.get("update")
        if update_event is not None:
            engine.on(update_event, self._on_update)
        # Attach engine start event handler for named metrics
        if groups is None:
            groups = "default"
        groups = (groups,) if isinstance(groups, str) else groups
        if name:
            engine.on(Events.STARTED, self._on_start, args=(groups,))
    
    def __add__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x+y, args=(self, other))
    
    def __radd__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x+y, args=(other, self))
    
    def __sub__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x-y, args=(self, other))
    
    def __rsub__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x-y, args=(other, self))
    
    def __mul__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x*y, args=(self, other))
    
    def __rmul__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x*y, args=(other, self))
    
    def __truediv__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x/y, args=(self, other))
    
    def __rtruediv__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x/y, args=(other, self))

    def __pow__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x**y, args=(self, other))
    
    def __rpow__(self: "Metric[TensorT]", other: Numerical) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x**y, args=(other, self))
    
    def __getitem__(self: "Metric[TensorT]", key) -> "Metric[TensorT]":
        return LambdaMetric(lambda x, y: x[y], args=(self, key))
    
    def attr(self, name: str) -> "Metric[Any]":
        return LambdaMetric(getattr, args=(self, name))
    
    def call(self, name: str, *args: Any, **kwargs: Any) -> "Metric[Any]":
        def _call_helper(_self, *_args: Any, **_kwargs: Any):
            return getattr(_self, name)(*_args, **_kwargs)

        return LambdaMetric(_call_helper, args=(self, *args), kwargs=kwargs)

class LambdaMetric(Metric[T]):
    __slots__ = ("f", "args", "kwargs")

    def __init__(self, f: Callable[..., T], *, args: Args = (), kwargs: Kwargs = {}):
        super().__init__()

        self.f = f
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    def compute(self) -> T:
        args = self.args
        # "Unlift" arguments by computing the value of dependent metrics
        if args:
            unlift_args = [
                arg.compute() if isinstance(arg, Metric) else arg \
                for arg in args
            ]
        # Provide engine state to function if no arguments are specified
        else:
            unlift_args = [self._engine.state]

        # "Unlift" keyword arguments by computing the value of dependent metrics
        unlift_kwargs = {
            key: (arg.compute() if isinstance(arg, Metric) else arg) \
            for key, arg in self.kwargs.items()
        }

        # Compute metric
        return self.f(*unlift_args, **unlift_kwargs)

    def attach(self, engine: Engine, name: str, groups: Union[None, str, Iterable[str]] = None):
        attach_dependencies(engine, (
            arg for arg in chain(self.args, self.kwargs) \
            if isinstance(arg, Metric)
        ))
        
        super().attach(engine, name, groups)

class StateMetric(Metric[_StateT]):
    __slots__ = ()
    
    def compute(self) -> _StateT:
        state = self._engine.state

        # Engine state is non-null when the engine is running
        assert state is not None

        return state

def attach_dependencies(engine: Engine, metrics: Iterable[Metric[Any]]):
    for metric in metrics:
        metric.attach(engine, "")
