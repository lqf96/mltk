from typing import Callable, Dict, Iterable, List, Mapping, Optional, Union, overload
from typing_extensions import Protocol
from mltk.types import Kwargs

from itertools import count

import torch as th
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import Optimizer as TorchOptim
from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    "Func",
    "Optimizer",
    "StopCond",
    "run_n_iters"
]

# A mathmatical function to be minimized
Func = Callable[[], th.Tensor]

class StopCond(Protocol):
    """ A callable protocol that determines whether to stop optimization or not. """
    def __call__(self, *, iterations: int, objectives: th.Tensor) -> th.Tensor: ...

# Group of parameters to be optimized
_Params = Union[th.Tensor, Iterable[th.Tensor], Module]
# PyTorch optimizer factory
_OptimFactory = Callable[..., TorchOptim]
# Learning rate scheduler factory
_LRSchedFactory = Callable[..., _LRScheduler]

_DEFAULT_OPTIM_GROUPS: Dict[str, Kwargs] = {"_default": {}}

class Optimizer():
    """\
    A helper class that wraps PyTorch optimizers and learning rate schedulers, as well as
    providing high-level optimization functionalities.

    Attributes:
        optim_factory: PyTorch optimizer factory.
        optim_params: Shared PyTorch optimizer settings for all parameter groups.
        optim_groups: Optimizer parameter groups and their group-specific settings.
        lr_sched_factory: Optional PyTorch learning rate scheduler factory.
        lr_sched_params: Learning rate scheduler settings.
    """

    __slots__ = (
        "optim_factory",
        "optim_params",
        "lr_sched_factory",
        "optim_groups",
        "lr_sched_params",
        "_attached_params",
        "_optim",
        "_lr_sched"
    )

    def __init__(self, optim_factory: _OptimFactory, optim_params: Kwargs = {},
        optim_groups: Mapping[str, Kwargs] = _DEFAULT_OPTIM_GROUPS,
        lr_sched_factory: Optional[_LRSchedFactory] = None,
        lr_sched_params: Kwargs = {}):
        self.optim_factory = optim_factory
        self.optim_params = dict(optim_params)
        self.optim_groups = dict(optim_groups)
        self.lr_sched_factory = lr_sched_factory
        self.lr_sched_params = dict(lr_sched_params)

        # Attached parameters
        self._attached_params: Dict[str, List[th.Tensor]] = {}
        # PyTorch optimizer
        self._optim: Optional[TorchOptim] = None
        # PyTorch learning rate scheduler (optional)
        self._lr_sched: Optional[_LRScheduler] = None

    @overload
    def attach_params(self, params: _Params, group_name: str = "_default") -> "Optimizer": ...

    @overload
    def attach_params(self, **params_kwargs: _Params) -> "Optimizer": ...

    def attach_params(self, params: Optional[_Params] = None, group_name: str = "_default",
        **params_kwargs: _Params) -> "Optimizer":
        if params is not None:
            # No other parameters can be provided beyond the given group
            if params_kwargs:
                raise ValueError("parameters cannot be provided for other groups")
            params_kwargs = {group_name: params}
        
        optim_groups = self.optim_groups
        attached_params = self._attached_params
        
        for group_name, new_params in params_kwargs.items():
            # Check existance of group
            if group_name not in optim_groups:
                raise KeyError(f"non-existant parameter group: {group_name}")
            
            # Convert or wrap parameters as an iterable
            if isinstance(new_params, th.Tensor):
                new_params = (new_params,)
            elif isinstance(new_params, Module):
                new_params = new_params.parameters()
            # Append new parameters to the group
            group_params = attached_params.setdefault(group_name, [])
            group_params.extend(new_params)
        
        return self
    
    def step(self, objective: th.Tensor, retain_graph: bool = False, create_graph: bool = False,
        clip_norm=None) -> "Optimizer":
        optim = self._optim
        lr_sched = self._lr_sched
        # Do nothing if optimizer does not exist
        if optim is None:
            pass

        # Back-propagate gradients from the objective
        objective.backward(retain_graph=retain_graph, create_graph=create_graph)
        # Clip gradients
        if clip_norm is not None:
            for group in optim.param_groups:
                clip_grad_norm_(group["params"], clip_norm)
        
        # Step the optimizer
        optim.step()
        # Step the LR scheduler (this must happen after the optimizer step)
        if lr_sched is not None:
            lr_sched.step()
        # Zero gradient of parameters if computation graph is not retained
        if not retain_graph and not create_graph:
            optim.zero_grad()
        
        return self
    
    def zero_grad(self) -> "Optimizer":
        optim = self._optim
        # Zero gradient of the optimizer if it exists
        if optim is not None:
            optim.zero_grad()
        
        return self

    def reset(self) -> "Optimizer":
        optim_groups = self.optim_groups
        lr_sched_factory = self.lr_sched_factory
        attached_params = self._attached_params

        # Parameters must be provided for all groups
        if len(attached_params)<len(optim_groups):
            raise ValueError("parameters must be provided for all groups")
        # Create PyTorch optimizer from parameters and settings
        self._optim = optim = self.optim_factory((
            {"params": attached_params[group_name], **group_settings} \
            for group_name, group_settings in optim_groups.items()
        ), **self.optim_params)
        # Create PyTorch learning rate scheduler
        if lr_sched_factory:
            self._lr_sched = lr_sched_factory(optim, **self.lr_sched_params)
        
        return self

    def minimize(self, func: Func, stop_cond: StopCond, mask: Optional[th.Tensor] = None
        ) -> th.Tensor:
        # Complete flags
        completed = None if mask is None else ~mask.bool()
        zero_scalar = None

        for i in count():
            objectives = func()

            # Initialize completed flags
            if completed is None:
                completed = th.zeros_like(objectives, dtype=th.bool)
            # Initialize zero scalar
            if zero_scalar is None:
                zero_scalar = objectives.new_zeros(())
            
            # Check validity of objectives
            if objectives.device!=completed.device:
                raise ValueError(
                    f"Expect objectives on device {completed.device}, got {objectives.device}"
                )
            elif objectives.shape!=completed.shape:
                raise ValueError(
                    f"Expect objectives of shape {completed.shape}, got {objectives.shape}"
                )
            
            # Check stop condition
            completed |= stop_cond(iterations=i, objectives=objectives.detach())
            if completed.all():
                return objectives.detach()
            
            # Compute a single step objective
            step_objective = th.where(completed, zero_scalar, objectives).sum()
            step_objective /= (~completed).sum()
            # Perform a single optimization step
            self.step(step_objective)

        # Count iterator is unbounded
        assert False

def run_n_iters(n_iterations: int) -> StopCond:
    """\
    Run minimization for given number of iterations.

    Args:
        n_iterations: Number of iterations to run.
    
    Returns:
        A callable stop condition that stops minimization after given number of iterations.
    """
    def n_iters_cond(iterations: int, **kwargs) -> th.Tensor:
        return th.tensor(iterations>n_iterations)
    
    return n_iters_cond
