from typing import List, Iterable, Optional, Union, Callable

from functools import partial
from itertools import count, chain

import numpy as np
import torch as th
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mltk import util

__all__ = [
    "DualOptimizer",
    "DiffLessThan",
    "Newton",
    "stop_after_n_iter",
    "optimize",
    "newton_solve"
]

def stop_after_n_iter(n_iteration: int):
    """ Stop optimization after given number of iterations. """
    return lambda iteration, **kwargs: iteration>=n_iteration

class DiffLessThan(object):
    def __init__(self, threshold, show=False):
        ## Threshold
        self.threshold = threshold
        ## Last values
        self._last_values = None
        self._count = 0
        self.show = show
    def __call__(self, objectives, **kwargs):
        threshold = th.tensor(self.threshold, device=objectives.device)
        last_values = self._last_values
        self._last_values = objectives.detach().clone()
        self._count += 1
        # Compute and compare difference
        if last_values is None:
            return th.tensor(False, device=objectives.device)
        else:
            difference = objectives-last_values
            return th.abs(objectives-last_values)<threshold

class DualOptimizer(object):
    def __init__(self, params_optimizer_factory: Optimizer = SGD,
        multipliers_optimizer_factory: Optimizer = SGD,
        params_lr_sched_factory: Optional[_LRScheduler] = None,
        multipliers_lr_sched_factory: Optional[_LRScheduler] = None,
        device: Union[th.device, str] = "cpu"):
        ## Parameters backward function
        self.params_backward_func = params_backward_func
        ## Multipliers backward function
        self.multipliers_backward_func = multipliers_backward_func
        ## Parameters optimizer factory
        self.params_optimizer_factory = params_optimizer_factory
        ## Multipliers optimizer factory
        self.multipliers_optimizer_factory = multipliers_optimizer_factory
        ## Parameters learning rate scheduler factory
        self.params_lr_sched_factory = params_lr_sched_factory
        ## Multipliers learning rate scheduler factory
        self.multipliers_lr_sched_factory = multipliers_lr_sched_factory
        ## PyTorch device for optimization
        self.device = th.device(device)
    def _dual_func(self, *params, with_objective=False):
        # Compute function value
        dual_values = self._func(*params)
        # Add equality and inequality constraint items
        for multipliers, constraint_func in chain(
            zip(self._eq_multipliers, self._eq_constraints),
            zip(self._ieq_multipliers, self._ieq_constraints)
        ):
            dual_values = dual_values+multipliers*constraint_func(*params)
        # Return dual values
        return dual_values
    def optimize(self, f, init_params, params_stop_cond, multipliers_stop_cond,
        eq_constraints=[], ieq_constraints=[], params_lr: Optional[float] = None,
        multipliers_lr: float = 0.2, params_optimizer_params: dict = {},
        multipliers_optimizer_params: dict = {}, params_lr_sched_params: dict = {},
        multipliers_lr_sched_params: dict = {}):
        device = self.device
        # Store function, equality and inequality constraints
        self._func = f
        self._eq_constraints = eq_constraints
        self._ieq_constraints = ieq_constraints
        # Initialize parameters
        params = [th.as_tensor(param, device=self.device) for param in init_params]
        # Call function to obtain objective shape
        objectives_shape = f(*params).shape
        # Initialize multipliers
        self._eq_multipliers = eq_multipliers = th.ones(
            (len(eq_constraints), *objectives_shape), requires_grad=True, device=device
        )
        self._ieq_multipliers = ieq_multipliers = th.ones(
            (len(ieq_constraints), *objectives_shape), requires_grad=True, device=device
        )
        # Multipliers optimizer and learning rate scheduler
        multipliers_optimizer = self.multipliers_optimizer_factory(
            [eq_multipliers, ieq_multipliers],
            lr=multipliers_lr,
            **multipliers_optimizer_params
        )
        if self.multipliers_lr_sched_factory is None:
            multipliers_lr_sched = None
        else:
            multipliers_lr_sched = self.multipliers_lr_sched_factory(
                multipliers_optimizer, **multipliers_lr_sched_params
            )
        # Complete flags and zero scalar tensor
        complete_flags = None
        zero_scalar = th.tensor(0, dtype=params[0].dtype, device=device)
        # Dual optimization loop
        for i in count():
            # Fix multipliers and optimize parameters
            params, objectives = optimize(
                self._dual_func,
                init_params=params,
                stop_cond=params_stop_cond,
                lr=params_lr,
                optimizer_factory=self.params_optimizer_factory,
                lr_sched_factory=self.params_lr_sched_factory,
                optimizer_params=params_optimizer_params,
                lr_sched_params=params_lr_sched_params,
                device=device,
                mask=complete_flags,
                backward_func=self.params_backward_func
            )
            # Initialize complete flags
            if complete_flags is None:
                complete_flags = th.zeros_like(objectives, dtype=th.bool, device=device)
            # Compute dual objectives
            dual_objectives = self._dual_func(*params)
            # Perform single optimization step
            run_optimization_step(
                objectives=-th.where(complete_flags, zero_scalar, dual_objectives),
                optimizer=multipliers_optimizer,
                lr_sched=multipliers_lr_sched,
                backward_func=self.multipliers_backward_func
            )
            # Constrain inequality multipliers
            ieq_multipliers = th.max(ieq_multipliers.detach(), zero_scalar)
            ieq_multipliers.requires_grad_()
            # Check stop condition
            complete_flags |= multipliers_stop_cond(
                iteration=i, params=params, eq_multipliers=eq_multipliers,
                ieq_multipliers=ieq_multipliers, objectives=dual_objectives
            )
            if th.all(complete_flags):
                break
        # Remove temporary members
        del self._func
        del self._eq_constraints
        del self._ieq_constraints
        del self._eq_multipliers
        del self._ieq_multipliers
        # Return optimized parameters and function value
        return params, f(*params)
