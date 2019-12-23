import numpy as np

__all__ = [
    "exponential_decay",
    "reciprocal_decay",
    "reciprocal_sqrt_decay",
    "dcm_decay",
    "apply_decay"
]

def exponential_decay(gamma: float, epoch: int, step_size: int = 1):
    return gamma**(epoch//step_size)

def reciprocal_decay(a: float, epoch: int, b: float = 1, step_size: int = 1):
    return 1/(a*(epoch//step_size)+b)

def reciprocal_sqrt_decay(a: float, epoch: int, b: float = 1, step_size: int = 1):
    return 1/(a*np.sqrt(epoch//step_size)+b)

def dcm_decay(tau: float, epoch: int, step_size: int = 1):
    n = epoch//step_size
    # DCM decay formula
    return 1/(1+n*n/(tau+n))

def apply_decay(value, decay_func, epoch):
    # Compute decay factor
    decay_factor = decay_func(epoch=epoch) if decay_func is not None else 1
    # Decay value
    return value*decay_factor
