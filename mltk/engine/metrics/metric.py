from typing import Dict, Generic, Optional
from mltk.types import T

from abc import ABC, abstractmethod

from ..engine import Engine, State
from ..events import AbstractEvents, Events

Triggers = Dict[str, AbstractEvents]

# TODO: Reset moving average
DEFAULT_TRIGGER_EVENTS: Triggers = {
    "update": Events.EPOCH_COMPLETED,
    "record": Events.EPOCH_COMPLETED
}

class Metric(ABC, Generic[T]):
    __slots__ = ("triggers", "_engine")

    def __init__(self, triggers: Triggers):
        super().__init__()

        ## Trigger points of the metric
        self.triggers = triggers
        ## Engine that the metric is attached to
        self._engine: Optional[Engine] = None

    def _on_start(self, engine: Engine, name: str):
        engine.metrics[name] = self

    def _on_reset(self, _: Engine):
        self.reset()

    def _on_update(self, _: Engine):
        self.update()

    def _on_record(self, engine: Engine, name: str):
        engine.state.metrics[name] = self.compute()

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

    def attach(self, engine: Engine, name: str):
        if self.attached:
            # Attempt to attach metric to a second engine
            if engine!=self._engine:
                raise ValueError("cannot attach metric to a different engine")
            # Do nothing as the metric is already attached to engine
            else:
                return     
        
        triggers = self.triggers
        self._engine = engine
        
        # Attach optional reset event handler
        reset_event = triggers.get("reset")
        if reset_event is not None:
            engine.on(reset_event, self._on_reset)
        # Attach optional update event handler
        update_event = triggers.get("update")
        if update_event is not None:
            engine.on(update_event, self._on_update)
        # Attach record event handler for named metric
        # (Empty event name represents an internal metric)
        if name!="":
            engine.on(Events.STARTED, self._on_start, args=(name,))
            engine.on(triggers["record"], self._on_record, args=(name,))
    
    def attach_dependency(self, engine: Engine):
        self.attach(engine, "")

class Root(Metric[State]):
    __slots__ = ()

    def __init__(self, record_trigger: AbstractEvents = Events.EPOCH_COMPLETED):
        super().__init__(triggers={"record": record_trigger})
    
    def compute(self) -> State:
        state = self._engine.state

        # Engine state is non-null when the engine is running
        assert state is not None

        return state
