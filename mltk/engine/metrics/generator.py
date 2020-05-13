from typing import Any, Callable, Generator, Optional
from mltk.types import Args, Kwargs, T, U

import functools

from .metric import Metric, Triggers

GenMetricFunc = Callable[..., Generator[T, U, None]]

class GeneratorMetric(Metric[T]):
    __slots__ = ("f", "src", "args", "kwargs", "_gen", "_value")

    def __init__(self, f: GenMetricFunc[T, U], src: Metric[U], triggers: Triggers, *,
        args: Args = (), kwargs: Kwargs = {}):
        super().__init__(triggers=triggers)
        
        ## Pure generator function (no side effect beyond internal state change)
        self.f = f
        ## Source (inner) metric
        self.src = src
        ## Arguments of the function
        self.args = tuple(args)
        ## Keyword arguments of the function
        self.kwargs = dict(kwargs)

        # Reset metric state
        self.reset()

    @classmethod
    def wraps(cls, triggers: Triggers):
        def gen_metric_decorator(f: GenMetricFunc[T, U]) -> Callable[..., GeneratorMetric[T]]:
            @functools.wraps(f)
            def gen_metric_factory(src: Metric[U], *args: Any, **kwargs: Any
                ) -> GeneratorMetric[T]:
                return cls(f, src, triggers, args=args, kwargs=kwargs)
            
            return gen_metric_factory
        
        return gen_metric_decorator

    def reset(self):
        gen_old: Optional[Generator[T, U, None]] = getattr(self, "_gen", None)
        # Notify generator to clean up during reset
        if gen_old is not None:
            try:
                next(gen_old)
            except StopIteration:
                pass
        
        ## Generator for computing the metric
        gen = self._gen = self.f(*self.args, **self.kwargs)
        # Initialize generator
        next(gen)

    def update(self):
        # Compute metric with current generator
        self._value = self._gen.send(self.src.compute())

    def compute(self) -> T:
        return self._value

    def attach(self, engine, name):
        # Attach source metric
        self.src.attach_dependency(engine)
        # Attach self
        super().attach(engine, name)
