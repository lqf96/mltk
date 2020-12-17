from typing import Iterable, Optional, Union
from mltk.types import TensorT

from ..engine import Engine
from .metric import Metric, Triggers, attach_dependencies

__all__ = [
    "Average",
    "RunningAverage"
]

class Average(Metric[TensorT]):
    __slots__ = ("src", "_count", "_avg")

    def __init__(self, src: Metric[TensorT], triggers: Triggers):
        super().__init__(triggers=triggers)
        ## Source (inner) metric
        self.src = src

        # Reset metric state
        self.reset()

    def reset(self):
        ## Average value of inner metric
        self._avg: Optional[TensorT] = None
        ## Value count
        self._count: int = 0
    
    def update(self):
        avg = self._avg
        count = self._count
        value = self.src.compute()

        # Initialize average value
        if avg is None:
            self._avg = value
        # Update average value
        else:
            count_next = count+1
            self._avg = avg*(count/count_next)+value/count_next
        # Update value count
        self._count += 1
    
    def compute(self) -> TensorT:
        # Metric update is always performed before compute, so average value cannot be empty
        assert self._avg is not None

        return self._avg
    
    def attach(self, engine: Engine, name: str, groups: Union[None, str, Iterable[str]] = None):
        # Attach source metric
        attach_dependencies(engine, (self.src,))
        # Attach self
        super().attach(engine, name)

class RunningAverage(Metric[TensorT]):
    __slots__ = ("src", "decay", "_avg")

    def __init__(self, src: Metric[TensorT], decay: float, triggers: Triggers):
        super().__init__(triggers=triggers)
        ## Source (inner) metric
        self.src = src
        ## Decay of running average
        self.decay = decay

        # Reset metric state
        self.reset()
    
    def reset(self):
        ## Average value of inner metric
        self._avg = None
    
    def update(self):
        decay = self.decay
        avg = self._avg
        value = self.src.compute()

        # Initialize average value
        if avg is None:
            self._avg = value
        # Update average value
        else:
            self._avg = decay*avg+(1-decay)*value
    
    def compute(self) -> TensorT:
        # Metric update is always performed before compute, so average value cannot be empty
        assert self._avg is not None

        return self._avg

    def attach(self, engine: Engine, name: str, groups: Union[None, str, Iterable[str]] = None):
        # Attach source metric
        attach_dependencies(engine, (self.src,))
        # Attach self
        super().attach(engine, name)
