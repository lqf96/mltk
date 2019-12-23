from typing import Any, Callable, Generator
from mltk.typing import Args, Kwargs

import functools, inspect
from abc import abstractmethod

from ignite.metrics import Metric, MetricsLambda

from mltk.util.internal import _NO_VALUE

__all__ = [
    "GeneratorMetric",
    "OneshotMetric"
]

class GeneratorMetric(Metric):
    def __init__(self, *base_args: Any, gen_func: Callable[..., Generator[None, Any, Any]],
        args: Args = (), kwargs: Kwargs = {}, **base_kwargs: Any):
        super().__init__(*base_args, **base_kwargs)
        ## Generator function
        self.gen_func = gen_func
        ## Arguments for generator function
        self.args = args
        ## Keyword arguments for generator function
        self.kwargs = kwargs

    @classmethod
    def wraps(cls, *base_args: Any, **base_kwargs: Any):
        def _decorator(gen_func: Callable[..., Generator[None, Any, Any]]):
            @functools.wraps(gen_func)
            def _metric_factory(*args: Any, **kwargs: Any):
                return cls(*base_args, gen_func, args, kwargs, **base_kwargs)
            return _metric_factory
        return _decorator

    def reset(self):
        # Create new generator for epoch
        self._gen = gen = self.gen_func(*self.args, **self.kwargs)
        # Initialize generator
        next(gen)

    def update(self, output: Any):
        # Resume generator with output
        self._gen.send(output)

    def compute(self):
        gen = self._gen

        # Resume generator and signal completion
        try:
            result = next(gen)
            gen.close()
        # Generator terminated
        except StopIteration as e:
            result = e.value

        return result

class OneshotMetric(Metric):
    def __init__(self, *base_args: Any, func: Callable[..., Any], args: Args = (),
        kwargs: Kwargs = {}, trigger_event: Any = Events.ITERATION_COMPLETED,
        **base_kwargs: Any):
        super().__init__(*base_args, trigger_events=trigger_event, **base_kwargs)
        ## Metric function
        self.func = func
        ## Arguments for metric function
        self.args = args
        ## Keyword srguments for metric function
        self.kwargs = kwargs

    @classmethod
    def wraps(cls, *base_args: Any, **base_kwargs: Any):
        def _decorator(func: Callable[..., Any]):
            @functools.wraps(gen_func)
            def _metric_factory(*args: Any, **kwargs: Any):
                return cls(*base_args, func, args, kwargs, **base_kwargs)
            return _metric_factory
        return _decorator

    def update(self, output):
        # Update called more than once each round
        if self._output is not _NO_VALUE:
            raise ValueError("update called more than once each round")

        # Store output
        self._output = output

    def compute(self):
        # Get and clear output
        output = self._output
        self._output = _NO_VALUE

        # Compute oneshot metric
        return self.func(output, *self.args, **self.kwargs)
