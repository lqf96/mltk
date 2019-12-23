from typing import Any, Optional, Iterable, Set

from logging import Logger, getLogger

from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, CallableEvents, Events

__all__ = [
    "MetricsPrinter",
    "MetricsWriter"
]

def _check_add_event_handler(engine: Engine, event: CallableEvents, handler,
    *args: Any, **kwargs: Any):
    if not engine.has_event_handler(handler, event):
        engine.add_event_handler(event, handler, *args, **kwargs)

class MetricsPrinter(object):
    def __init__(self, interval: int, event: CallableEvents = Events.EPOCH_COMPLETED,
        logger: Optional[Logger] = None, include_metrics: Iterable[str] = (),
        exclude_metrics: Iterable[str] = ()):
        # Default logger for printing
        if logger is None:
            logger = getLogger(f"{__name__}.{self.__class__.__name__}")
        # Include and exclude metrics cannot be specified at same time
        if include_metrics and exclude_metrics:
            raise ValueError(
                "Include and exclude metrics cannot be specified at same time"
            )

        ## Printing interval
        self.interval = interval
        ## Printing event
        self.event = event
        ## Logger for printing
        self.logger = logger
        ## Metrics to be included
        self.include_metrics = set(include_metrics)
        ## Metrics to be excluded
        self.exclude_metrics = set(exclude_metrics)

        ## Engine-metric groups mapping
        self._engine_metric_groups = {}

    def _on_engine_start(self, engine: Engine):
        engine_state = engine.state
        # All metrics to be printed
        metrics = self.include_metrics or set(engine_state.metrics_meta.keys())
        metrics = metrics.difference(self.exclude_metrics)
        # Metrics metadata and event-state attribute mapping
        event_attr = engine_state.event_to_attr
        metrics_meta = engine_state.metrics_meta

        metric_groups = self._engine_metrics_groups[engine] = {}
        # Build metric groups mapping
        for metric in metrics:
            # Get metric event and corresponding state attribute
            metric_event = metrics_meta[metric]["trigger_events"]["completed"]
            metric_attr = event_attr[metric_event]
            # Insert metric name to metric group
            group = metric_groups.setdefault(metric_attr, [])
            group.append(metric)

    def _on_engine_completed(self, engine: Engine):
        # Remove metric groups mapping for engine
        del self._engine_metric_groups[engine]

    def _on_event(self, engine: Engine):
        engine_state = engine.state
        # TODO: Use logger for printing
        metrics = engine_state.metrics
        metric_groups = self._engine_metrics_groups[engine]
        
        # Print for each metric group
        for metric_attr, metric_names in metric_groups.items():
            attr_value = getattr(engine_state, metric_attr)
            # Print metric group title
            print(f"* {metric_attr}: {attr_value}")

            # Print each metric in group
            for metric_name in metric_names:
                print(f"  - {metric_name}: {metrics[metric_name]}")

    def attach(self, engine: Engine, name: str):
        # Add engine started handler
        _check_add_event_handler(engine, Events.STARTED, self._on_engine_start)
        # Add printing event handler
        _check_add_event_handler(engine, self.event, self._on_event)
        # Add engine completed handler
        _check_add_event_handler(engine, Events.COMPLETED, self._on_engine_completed)

class MetricsWriter(object):
    def __init__(self, writer: SummaryWriter, events: Iterable[CallableEvents] = (),
        include_metrics: Iterable[str] = (), exclude_metrics: Iterable[str] = (),
        prefix_by_attr: bool = True):
        # Include and exclude metrics cannot be specified at same time
        if include_metrics and exclude_metrics:
            raise ValueError(
                "Include and exclude metrics cannot be specified at same time"
            )

        ## Summary writer
        self.writer = writer
        ## Triggering events for writer
        self.events = set(events)
        ## Metrics to be included
        self.include_metrics = set(include_metrics)
        ## Metrics to be excluded
        self.exclude_metrics = set(exclude_metrics)
        ## Prefix metric name by attribute
        self.prefix_by_attr = prefix_by_attr

        ## Engine-metric groups mapping
        self._engine_metric_groups = {}

    def _on_engine_start(self, engine: Engine):
        engine_state = engine.state
        # All metrics to be printed
        metrics = self.include_metrics or set(engine_state.metrics_meta.keys())
        metrics = metrics.difference(self.exclude_metrics)
        # All triggering events
        events = self.events
        # Metrics metadata
        metrics_meta = engine_state.metrics_meta

        metric_groups = self._engine_metrics_groups[engine] = {}
        # Build metric groups mapping
        for metric in metrics:
            # Skip metric if its triggering events is not included
            metric_event = metrics_meta[metric]["trigger_events"]["completed"]
            if events and metric_event not in events:
                continue
            # Insert metric name to metric group
            group = metric_groups.setdefault(metric_event, [])
            group.append(metric)

        # Register event handlers for writing
        for event in metric_groups.keys():
            # Prepend event attribute to metric name
            metric_prefix = engine_state.event_to_attr(event)+"/" \
                if self.prefix_by_attr else ""
            _check_add_event_handler(engine, self._on_event, event, metric_prefix)

    def _on_engine_completed(self, engine: Engine):
        # Remove metric groups mapping for engine
        del self._engine_metric_groups[engine]

    def _on_event(self, engine: Engine, event: CallableEvents, metric_prefix: str):
        engine_state = engine.state
        # Tensorboard summary writer
        writer = self.writer

        # Get metric group and all metrics values
        metrics = engine_state.metrics
        group = self._engine_metric_groups[engine][event]
        # Add data point for each metric in group
        for metric_name in group:
            writer.add_scalar(metric_prefix+metric_name, float(metrics[metric_name]))

    def attach(self, engine: Engine):
        # Add engine started handler
        _check_add_event_handler(engine, Events.STARTED, self._on_engine_start)
        # Add engine completed handler
        _check_add_event_handler(engine, Events.COMPLETED, self._on_engine_completed)
