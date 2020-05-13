from logging import warn
from typing import Any, Callable, Optional, Tuple
from mltk.types import Args, Kwargs, StrDict, T

import warnings
from itertools import chain

from ..engine import Engine
from ..events import AbstractEvents, Events
from .metric import Metric

def _infer_record_trigger(args: Tuple[Any, ...], kwargs: StrDict) -> AbstractEvents:
    default_trigger = Events.EPOCH_COMPLETED
    record_trigger = None

    for arg in chain(args, kwargs.values()):
        # Ignore non-metric arguments
        if not isinstance(arg, Metric):
            continue
        
        arg_trigger = arg.triggers["record"]
        # Infer record trigger from the first argument
        if record_trigger is None:
            record_trigger = arg_trigger
        # Inconsistent record triggers among arguments
        elif record_trigger!=arg_trigger:
            warnings.warn(
                "metrics from arguments and keyword arguments have inconsistent "
                f"record triggers; falling back to \"{default_trigger}\" event.",
                RuntimeWarning
            )
            return default_trigger

    # Arguments and keyword arguments do not contain any metrics
    if record_trigger is None:
        warnings.warn(
            "no metrics found in arguments and keyword arguments; "
            f"record trigger set to \"{default_trigger}\" event.",
            RuntimeWarning
        )
        return default_trigger
    
    return record_trigger

class LambdaMetric(Metric[T]):
    __slots__ = ("f", "args", "kwargs")

    def __init__(self, f: Callable[..., T], *, args: Args = (), kwargs: Kwargs = {},
        record_trigger: Optional[AbstractEvents] = None):
        ## Pure function for computing metric
        self.f = f
        ## Arguments of the function
        self.args = tuple(args)
        ## Keyword arguments of the function
        self.kwargs = dict(kwargs)

        # Infer record triggers from arguments if not specified
        if record_trigger is None:
            record_trigger = _infer_record_trigger(self.args, self.kwargs)
        
        # Initialize base class
        super().__init__(triggers={"record": record_trigger})

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

    def attach(self, engine: Engine, name: str):
        # Attach dependent metrics in function arguments
        for arg in chain(self.args, self.kwargs.values()):
            if isinstance(arg, Metric):
                arg.attach_dependency(engine)
        
        # Attach self
        super().attach(engine, name)
