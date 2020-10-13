from typing import Any, Iterable, Optional, SupportsFloat, SupportsIndex, Tuple, Union, \
    overload
from mltk.types import Shape, Device, ModuleT

from contextlib import contextmanager

import numpy as np
import torch as th
from torch.nn import Module
from torch.distributions import LowRankMultivariateNormal

__all__ = [
    "SLICE_ALL",
    "as_th_dtype",
    "as_size",
    "derive_rand",
    "force_float",
    "isscalar",
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

def as_th_dtype(dtype: Union[str, type, np.dtype]) -> th.dtype:
    return _DTYPE_MAP[np.dtype(dtype).name]

def as_size(shape: Shape) -> th.Size:
    shape = (int(shape),) if isinstance(shape, SupportsIndex) else shape
    return th.Size(shape)

def derive_rand(rand: th.Generator, device: Device) -> th.Generator:
    device = th.device(device)
    # Reuse existing random number generator for the same device
    if rand.device==device:
        return rand
    
    # Create and seed new random number generator
    seed_new = th.randint(_SEED_MAX, (), device=rand.device, generator=rand)
    rand_new = th.Generator(device)
    rand_new.manual_seed(int(seed_new))

    return rand_new

@overload
def force_float(dtype: th.dtype, target_dtype: Optional[th.dtype] = None) -> th.dtype: ...

@overload
def force_float(tensor: th.Tensor, target_dtype: Optional[th.dtype] = None) -> th.Tensor: ...

def force_float(source: Union[th.dtype, th.Tensor], target_dtype: Optional[th.dtype] = None
    ) -> Union[th.dtype, th.Tensor]:
    # Force floating point data type for given data type
    if isinstance(source, th.dtype):
        target_dtype = target_dtype or th.get_default_dtype()
        return target_dtype if source.is_floating_point else source
    # Force floating point data type for given tensor
    else:
        return source.to(dtype=force_float(source.dtype, target_dtype))

def isscalar(tensor: th.Tensor) -> bool:
    return tensor.ndim==0

def one_hot(labels: th.Tensor, n: int, dtype: Optional[th.dtype] = None) -> th.Tensor:
    return th.eye(n, dtype=dtype, device=labels.device)[labels]

def rand_bool(shape: Shape = (), p: SupportsFloat = 0.5, *, device: Device = "cpu",
    rand: th.Generator = th.default_generator) -> th.Tensor:
    size = as_size(shape)
    true_prob = float(p)
    device = th.device(device)

    rand_values = th.rand(size, device=rand.device, generator=rand)
    return (rand_values<true_prob).to(device=device)

@contextmanager
def use_rand(rand: th.Generator, **kwargs: Any):
    with th.random.fork_rng(devices=(rand.device,), **kwargs):
        # "Fork" with derived random state
        seed_fork = th.randint(_SEED_MAX, (), device=rand.device, generator=rand)
        th.random.manual_seed(int(seed_fork))
        
        yield
        
        # "Join" changes of the global random state
        seed_join = th.randint(_SEED_MAX, ())
        rand.manual_seed(int(seed_join))

def zip_params(*modules: ModuleT) -> Iterable[Tuple[th.Tensor, ...]]:
    return zip(*(module.parameters() for module in modules))
