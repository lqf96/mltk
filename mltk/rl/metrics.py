from mltk.types.gym import O, A, R

from mltk.engine import Events
from mltk.engine.metrics import GeneratorMetric
from .execution import EPISODE_STARTED, RLState

__all__ = [
    "episode_length",
    "episode_reward"
]

@GeneratorMetric.wraps(triggers={
    "reset": EPISODE_STARTED,
    "update": Events.ITER_COMPLETED
})
def episode_length():
    length = 0
    state = yield length # type: RLState[O, A, R]
    
    while True:
        # End of episode
        if state is None:
            return
        
        length += 1
        # Yield updated episode length
        state = yield length

@GeneratorMetric.wraps(triggers={
    "reset": EPISODE_STARTED,
    "update": Events.ITER_COMPLETED
})
def episode_reward(discount_factor: float = 1):
    retn = 0.
    state = yield retn # type: RLState[O, A, R]

    while True:
        # End of episode
        if state is None:
            return
        
        retn = discount_factor*retn+state.transition.reward
        # Yield updated (discounted) return
        state = yield retn
