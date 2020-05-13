from typing import Callable, Optional
from mltk.types import TensorT

import math

__all__ = [
    "DecayFunc",
    "exponential_decay",
    "reciprocal_decay",
    "reciprocal_sqrt_decay",
    "dcm_decay",
    "apply_decay"
]

DecayFunc = Callable[[int], float]

def exponential_decay(gamma: float, step_size: int = 1) -> DecayFunc:
    return lambda count: gamma*(count//step_size)

def reciprocal_decay(a: float, b: float = 1, step_size: int = 1) -> DecayFunc:
    return lambda count: 1/(a*(count//step_size)+b)

def reciprocal_sqrt_decay(a: float, b: float = 1, step_size: int = 1) -> DecayFunc:
    return lambda count: 1/(a*math.sqrt(count//step_size)+b)

def dcm_decay(tau: float, epoch: int, step_size: int = 1) -> DecayFunc:
    def dcm_decay_func(count: int) -> float:
        n = count//step_size
        # DCM decay formula
        return 1/(1+n*n/(tau+n))
    
    return dcm_decay_func

def apply_decay(value: TensorT, decay_func: Optional[DecayFunc], count: int) -> TensorT:
    # Compute decay factor
    decay_factor = decay_func(count) if decay_func else 1.
    # Return decayed value
    return value*decay_factor
