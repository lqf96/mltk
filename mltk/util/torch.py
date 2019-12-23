from typing import Any, Tuple
import math, functools
from itertools import repeat
from functools import reduce
from contextlib import contextmanager

import numpy as np
from numpy.random import RandomState
import torch as th
from torch.random import fork_rng, manual_seed

__all__ = [
    "ALL_INDICES",
    "use_rand",
    "softmax",
    "logsumexp",
    "tensor_prod",
    "ignore_div_by_zero",
    "isscalar",
    "unsqueeze",
    "cartesian",
    "rand_bool",
    "one_hot",
    "inverse"
]

# Constant for representing all indices when slicing
ALL_INDICES = ...

@contextmanager
def use_rand(rand: th.Generator, **kwargs: Any):
    # Sample seed from random number generator
    seed = th.randint(rand, (), generator=rand)
    # Fork current random state (both CPU and GPU)
    with th.random.fork_rng(**kwargs):
        th.random.manual_seed(seed)
        yield

def ignore_div_by_zero():
    """ Return a context manager that ignore divide-by-zero warnings. """
    return np.errstate(invalid="ignore", divide="ignore")

def tensor_prod(*tensors):
    """
    Compute the tensor product of multiple tensors.
    """
    n_tensors = len(tensors)
    # Raise error if no input tensors are provided
    if n_tensors==0:
        raise ValueError("No input tensors are provided")
    # Return the tensor itself if one input tensor is provided
    elif n_tensors==1:
        return tensors[0]
    else:
        elements_shape = None
        # Check shape of all input tensors
        for i, tensor in enumerate(tensors):
            tensor_shape = np.shape(tensor)
            # Elements shape of first tensor
            if elements_shape is None:
                elements_shape = tensor_shape[:-1]
            # Elements shape mismatch
            elif elements_shape!=tensor_shape[:-1]:
                expected_shape_repr = "("
                for size in elements_shape:
                    expected_shape_repr += str(size)+", "
                expected_shape_repr += "*)"
                # Raise shape mismatch exception
                raise ValueError("Expect tensor of shape {}, got {} for tensor {}".format(
                    expected_shape_repr, tuple(tensor_shape), i+1
                ))
        # Reshape input tensors for broadcasting
        reshaped_tensors = (
            unsqueeze(unsqueeze(tensor, -2, count=i), -1, count=n_tensors-i-1)
            for i, tensor in enumerate(tensors)
        )
        # Compute tensor product
        return reduce(np.multiply, reshaped_tensors)

def softmax(tensor, dim, t):
    raise NotImplementedError

def logsumexp(tensor, dim, t):
    raise NotImplementedError

def isscalar(tensor):
    return np.ndim(tensor)==0

def unsqueeze(tensor, axis, count=1):
    # Check whether dimension is legal or not
    tensor_dims = np.ndim(tensor)
    if axis>tensor_dims or axis<-tensor_dims-1:
        raise IndexError("Dimension {} out of range [{}, {}]".format(
            axis, -tensor_dims-1, tensor_dims
        ))
    # Convert dimension to non-negative integer
    if axis<0:
        axis += tensor_dims+1
    # Check count
    if count<0:
        raise IndexError("Dimension count should be positive integer, got {}".format(count))
    # No dimension expansion needs to be made
    elif count==0:
        return tensor
    # Compute new shape of tensor
    tensor_shape = np.shape(tensor) if tensor_dims>0 else np.empty(0, dtype=int)
    new_shape = np.insert(tensor_shape, axis, np.repeat(1, count))
    # Reshape and return tensor
    return np.reshape(tensor, tuple(new_shape))

def cartesian(*tensors):
    n_tensors = len(tensors)
    # All input tensors must have exact one dimension
    for tensor in tensors:
        if np.ndim(tensor)!=1:
            raise ValueError("All input tensors must have exactly one dimension")
    # Compute cartesian product
    result = np.stack(np.meshgrid(*tensors, indexing="ij"), -1)
    return np.reshape(result, (-1, n_tensors))

def rand_bool(shape=(), p=0.5, rand=default_rand):
    return rand.rand(*shape)>p

def one_hot(indices, size):
    return np.eye(size)[indices]

def inverse(tensor, replacement=np.nan, with_det=False):
    # Compute determinants of all matrices and find non-zero items
    determinants = np.linalg.det(tensor)
    nonzero_det_indices = np.nonzero(determinants)
    # Compute inverse for non-zero items
    inv = np.full_like(tensor, replacement)
    inv[nonzero_det_indices] = np.linalg.inv(tensor[nonzero_det_indices])
    # Return inverse and (optionally) determinants
    return (inv, determinants) if with_det else inv
