from typing import Optional

from logging import Logger, getLogger

from torch.utils.tensorboard import SummaryWriter

from ..engine import Engine, every
from ..events import AbstractEvents, Events

__all__ = [
    "MetricsLogger",
    "MetricsWriter"
]

class MetricsLogger:
    def __init__(self, interval: int = 1, event: AbstractEvents = Events.EPOCH_COMPLETED,
        logger: Optional[Logger] = None):
        self.interval = interval
        self.event = event
        self.logger = logger or getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _on_event(self, engine: Engine, group: str):
        logger = self.logger
        engine_state = engine.state
        metric_group = engine._metric_groups[group]

        event_attr = engine_state.event_to_attr[self.event]
        attr_value = getattr(engine_state, event_attr)
        # Print event attribute name and value
        logger.info(f"* {group} ({event_attr} {attr_value})")
        # Print each metric
        for metric in metric_group:
            logger.info(f"  - {metric._name}: {metric.compute()}")

    def attach(self, engine: Engine, group: str):
        engine.on(self.event, self._on_event, args=(group,), filter=every(self.interval))

class MetricsWriter:
    def __init__(self, log_dir: str, interval: int = 1,
        event: AbstractEvents = Events.EPOCH_COMPLETED):
        self.interval = interval
        self.event = event

        self._writer = SummaryWriter(log_dir)
    
    def _on_event(self, engine: Engine, group: str):
        writer = self._writer
        metric_group = engine._metric_groups[group]

        event_count = engine.state.count_for(self.event)

        for metric in metric_group:
            writer.add_scalar(
                f"{group}/{metric._name}",
                metric.compute(),
                event_count
            )

    def attach(self, engine: Engine, group: str):
        engine.on(self.event, self._on_event, args=(group,), filter=every(self.interval))
