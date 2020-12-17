from __future__ import annotations

from typing import Optional, Protocol, Union, overload
from collections.abc import Callable, Iterable, Mapping
from typing_extensions import TypeAlias
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
Func: TypeAlias = "Callable[[], th.Tensor]"

class StopCond(Protocol):
    """ A callable protocol that determines whether to stop optimization or not. """
    def __call__(self, *, iterations: int, objectives: th.Tensor) -> th.Tensor: ...

# Group of parameters to be optimized
_Params: TypeAlias = "Union[th.Tensor, Iterable[th.Tensor], Module]"
# PyTorch optimizer factory
_OptimFactory: TypeAlias = "Callable[..., TorchOptim]"
# Learning rate scheduler factory
_LRSchedFactory: TypeAlias = "Callable[..., _LRScheduler]"

_DEFAULT_OPTIM_GROUPS: dict[str, Kwargs] = {"default": {}}

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

    def __init__(self, optim_factory: _OptimFactory, optim_config: Kwargs = {},
        groups_config: Mapping[str, Kwargs] = _DEFAULT_OPTIM_GROUPS,
        lr_sched_factory: Optional[_LRSchedFactory] = None, lr_sched_config: Kwargs = {},
        clip_max_norm: Optional[float] = None, clip_norm_type: int = 2,
        clip_max_value: Optional[float] = None):
        self.optim_factory = optim_factory
        self.optim_config = dict(optim_config)
        self.groups_config = dict(groups_config)
        self.lr_sched_factory = lr_sched_factory
        self.lr_sched_config = dict(lr_sched_config)
        self.clip_max_norm = clip_max_norm
        self.clip_norm_type = clip_norm_type
        self.clip_max_value = clip_max_value

        # Bound parameter groups
        self._bound_groups: dict[str, list[th.Tensor]] = {}
        # PyTorch optimizer
        self._optim: Optional[TorchOptim] = None
        # PyTorch learning rate scheduler (optional)
        self._lr_sched: Optional[_LRScheduler] = None

    @overload
    def bind(self, group_name: str, *params: _Params) -> Optimizer: ...

    # TODO: signature of bind

    def bind(self, group_name: str, *params: _Params) -> Optimizer:
        # Check existance of group
        if group_name not in self.groups_config:
            raise KeyError(f"non-existant parameter group: {group_name}")

        group_params = self._bound_groups.setdefault(group_name, [])
        for new_params in params:
            # Convert or wrap parameters as an iterable
            if isinstance(new_params, th.Tensor):
                new_params = (new_params,)
            elif isinstance(new_params, Module):
                new_params = new_params.parameters()
            # Append new parameters to the group
            group_params.extend(new_params)
        
        return self
    
    def step(self, *objectives: th.Tensor, retain_graph: bool = False, create_graph: bool = False
        ) -> Optimizer:
        optim = self._optim
        lr_sched = self._lr_sched
        # Do nothing if optimizer does not exist
        if optim is None:
            return

        # Back-propagate gradients from the objective
        for objective in objectives:
            objective.backward(retain_graph=retain_graph, create_graph=create_graph)
        
        # Clip gradients for parameter groups
        for group in optim.param_groups:
            group_params = group["params"]
            # Clip by norm
            max_norm = group.get("clip_max_norm", self.clip_max_norm)
            if max_norm is not None:
                norm_type = group.get("clip_norm_type", self.clip_norm_type)
                clip_grad_norm_(group_params, max_norm, norm_type)
                continue
            # Clip by value
            max_value = group.get("clip_max_value", self.clip_max_value)
            if max_value is not None:
                clip_grad_value_(group_params, max_value)
                continue
        
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
        groups_config = self.groups_config
        lr_sched_factory = self.lr_sched_factory
        bound_groups = self._bound_groups

        # Parameters must be bound for all groups
        if len(bound_groups)<len(groups_config):
            raise ValueError("parameters must be bound for all groups")

        # Create PyTorch optimizer from parameters and settings
        self._optim = optim = self.optim_factory((
            {"params": bound_groups[group_name], **group_config} \
            for group_name, group_config in groups_config.items()
        ), **self.optim_config)
        # Create PyTorch learning rate scheduler
        if lr_sched_factory:
            self._lr_sched = lr_sched_factory(optim, **self.lr_sched_config)
        
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
