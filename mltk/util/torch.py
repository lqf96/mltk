from typing import Any, Iterable, SupportsFloat, SupportsIndex, Tuple
from mltk.types import Shape, Device, ModuleT

from contextlib import contextmanager

import numpy as np
import torch as th
from torch.distributions import LowRankMultivariateNormal

__all__ = [
    "SLICE_ALL",
    "as_th_dtype",
    "as_size",
    "as_device",
    "derive_rand",
    "force_float",
    "isscalar",
    "multivariate_normal_diag",
    "one_hot",
    "rand_bool",
    "use_rand",
    "zip_params"
]

# A mapping from NumPy data type names to PyTorch data type
_DTYPE_MAP = {
    "int8": th.int8,
    "int16": th.int16,
    "int32": th.int32,
    "int64": th.int64,
    "uint8": th.uint8,
    "bool": th.bool,
    "float16": th.float16,
    "float32": th.float32,
    "float64": th.float64,
    "complex64": th.complex64,
    "complex128": th.complex128
}
# PyTorch random seed maximum value
_SEED_MAX = 0x0100_0000_0000_0000

SLICE_ALL = slice(None)

def as_th_dtype(dtype: np.dtype) -> th.dtype:
    return _DTYPE_MAP[dtype.name]

def as_size(shape: Shape) -> th.Size:
    shape = (shape.__index__(),) if isinstance(shape, SupportsIndex) else shape
    return th.Size(shape)

def as_device(dev: Device) -> th.device:
    return dev if isinstance(dev, th.device) else th.device(dev)

def derive_rand(rand: th.Generator, device: Device) -> th.Generator:
    device = as_device(device)
    # Return existing random number generator for the same device
    if rand.device==device:
        return rand
    
    # Create and seed new random number generator
    rand_new = th.Generator(device)
    seed_new = th.randint(_SEED_MAX, (), device=rand.device, generator=rand)
    rand_new.manual_seed(seed_new.item())

    return rand_new

def force_float(dtype: th.dtype, target_dtype: th.dtype) -> th.dtype:
    return target_dtype if dtype.is_floating_point else dtype

def isscalar(tensor: th.Tensor) -> bool:
    return tensor.ndim==0

def multivariate_normal_diag(loc: th.Tensor, scale: th.Tensor) -> LowRankMultivariateNormal:
    dist = LowRankMultivariateNormal(
        loc=loc,
        cov_factor=th.zeros_like(scale).unsqueeze(-1),
        cov_diag=scale
    )
    dist.scale = dist.cov_diag
    
    return dist

def one_hot(labels: th.Tensor, n: int) -> th.Tensor:
    return th.eye(n, device=labels.device)[labels]

def rand_bool(shape: Shape = (), p: SupportsFloat = 0.5, *, device: Device = "cpu",
    rand: th.Generator = th.default_generator) -> th.Tensor:
    size = as_size(shape)
    true_prob = p.__float__()
    device = as_device(device)

    # Generate random values between 0 and 1
    return th.rand(size, device=device, generator=rand)<true_prob

@contextmanager
def use_rand(rand: th.Generator, **kwargs: Any):
    # Fork random number generator state
    with th.random.fork_rng(devices=(rand.device,), **kwargs):
        # Seed global random number generator state
        seed_fork = th.randint(_SEED_MAX, (), device=rand.device, generator=rand)
        th.random.manual_seed(seed_fork.item())
        
        # Return to code in scope
        yield
        
        # "Join" changes of the global random number generator state
        seed_join = th.randint(_SEED_MAX, ())
        rand.manual_seed(seed_join.item())

def zip_params(*modules: ModuleT) -> Iterable[Tuple[th.Tensor, ...]]:
    return zip(*(module.parameters() for module in modules))
