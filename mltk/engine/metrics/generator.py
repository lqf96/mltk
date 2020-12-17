from typing import Any, Callable, Generator, Iterable, Optional, Union
from mltk.types import Args, Kwargs, T, U

import functools

from ..engine import Engine
from .metric import Metric, Triggers, attach_dependencies

_MetricGen = Generator[T, Optional[U], None]
GenMetricFunc = Callable[..., _MetricGen[T, U]]

class GeneratorMetric(Metric[T]):
    __slots__ = ("f", "src", "args", "kwargs", "_gen", "_value")

    def __init__(self, f: GenMetricFunc[T, U], src: Metric[U], triggers: Triggers, *,
        args: Args = (), kwargs: Kwargs = {}):
        super().__init__(triggers=triggers)
        
        self.f = f
        self.src = src
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

        self._gen: Optional[_MetricGen[T, U]] = None

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
        gen_old = self._gen
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

    def attach(self, engine: Engine, name: str, groups: Union[None, str, Iterable[str]] = None):
        # Attach source metric
        attach_dependencies(engine, (self.src,))
        # Attach self
        super().attach(engine, name, groups)
