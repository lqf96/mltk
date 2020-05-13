from enum import Enum

__all__ = [
    "AbstractEvents",
    "Events"
]

class AbstractEvents():
    pass

class Events(AbstractEvents, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    ITER_STARTED = "iteration_started"
    ITER_COMPLETED = "iteration_completed"
