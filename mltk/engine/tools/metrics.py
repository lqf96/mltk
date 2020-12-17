from typing import Optional

from logging import Logger, getLogger

from torch.utils.tensorboard import SummaryWriter

from ..engine import Engine
from ..types import AbstractEventKind, Events

__all__ = [
    "MetricsLogger",
    "MetricsWriter"
]

class MetricsLogger:
    def __init__(self, event: AbstractEventKind = Events.EPOCH_COMPLETED,
        logger: Optional[Logger] = None):
        self.event = event
        self.logger = logger or getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _on_event(self, engine: Engine, group: str):
        logger = self.logger
        engine_state = engine.state
        metric_group = engine._metric_groups[group]

        event_attr = engine_state.event_attr_map[self.event.kind]
        attr_value = getattr(engine_state, event_attr)
        # Print event attribute name and value
        logger.info(f"* {group} ({event_attr} {attr_value})")
        # Print each metric
        for metric in metric_group:
            logger.info(f"  - {metric._name}: {metric.compute()}")

    def attach(self, engine: Engine, group: str):
        engine.on(self.event, self._on_event, args=(group,))

class MetricsWriter:
    def __init__(self, log_dir: str, event: AbstractEventKind = Events.EPOCH_COMPLETED):
        self.event = event

        self._writer = SummaryWriter(log_dir)
    
    def _on_event(self, engine: Engine, group: str):
        writer = self._writer
        metric_group = engine._metric_groups[group]
        if not metric_group:
            return

        event_count = engine.state.count_for(self.event.kind)

        for metric in metric_group:
            writer.add_scalar(
                f"{group}/{metric._name}",
                metric.compute(),
                event_count
            )

    def attach(self, engine: Engine, group: str):
        engine.on(self.event, self._on_event, args=(group,))
