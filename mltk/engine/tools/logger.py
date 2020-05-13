from typing import Iterable, List, Optional, Union

from enum import Enum
from logging import Logger, getLogger

from ..engine import Engine, AbstractEvents, Events, every

_MetricsGroup = Iterable[Union[str, AbstractEvents]]

__all__ = [
    "DefaultMetrics",
    "MetricsLogger"
]

class DefaultMetrics(Enum):
    NONE = "<no metric>"
    DEFAULT = "<default metrics>"
    ALL = "<all metrics>"

class MetricsLogger():
    __slots__ = (
        "interval",
        "event",
        "logger",
        "default_metrics",
        "include_metrics",
        "exclude_metrics",
        "_engine",
        "_log_metrics"
    )

    def __init__(self, interval: int = 1, event: AbstractEvents = Events.EPOCH_COMPLETED,
        logger: Optional[Logger] = None, default_metrics: DefaultMetrics = \
        DefaultMetrics.DEFAULT, include_metrics: _MetricsGroup = (),
        exclude_metrics: _MetricsGroup = ()):
        # Default logger for printing
        if logger is None:
            logger = getLogger(f"{__name__}.{self.__class__.__name__}")

        ## Logging interval
        self.interval = interval
        ## Logging event
        self.event = event
        ## Logger
        self.logger = logger
        ## Default metrics mode
        self.default_metrics = default_metrics
        ## Metrics to be included
        self.include_metrics = set(include_metrics)
        ## Metrics to be excluded
        self.exclude_metrics = set(exclude_metrics)

    def _on_start(self, engine: Engine):
        default_metrics = self.default_metrics
        include_metrics = self.include_metrics
        exclude_metrics = self.exclude_metrics

        # Include all metrics that record on the log event for default mode
        if default_metrics==DefaultMetrics.DEFAULT:
            include_metrics.add(self.event)
        
        log_metrics: List[str] = []
        # Get name of metrics to be logged
        for name, metric in engine.metrics.items():
            metric_trigger = metric.triggers["record"]

            # Include current metric by name or record trigger
            if default_metrics!=DefaultMetrics.ALL:
                if name not in include_metrics and metric_trigger not in include_metrics:
                    continue
            # Exclude current metric by name or record trigger
            if name in exclude_metrics or metric_trigger in exclude_metrics:
                continue

            log_metrics.append(name)
        
        self._log_metrics = log_metrics

    def _on_event(self, engine: Engine):
        logger = self.logger
        engine_state = engine.state
        metrics = engine_state.metrics

        event_attr = engine_state.event_to_attr[self.event]
        attr_value = getattr(engine_state, event_attr)
        # Print event attribute name and value
        logger.info(f"* {event_attr}: {attr_value}")
        # Print each metric
        for metric_name in self._log_metrics:
            logger.info(f"  - {metric_name}: {metrics[metric_name]}")

    def attach(self, engine: Engine, name: str):
        # Add engine started handler
        engine.on(Events.STARTED, self._on_start)
        # Add printing event handler
        engine.on(self.event, self._on_event, filter=every(self.interval))
