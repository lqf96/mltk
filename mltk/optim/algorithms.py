from typing import List, Iterable, Optional, Union, Callable

from functools import partial
from itertools import count, chain

import numpy as np
import torch as th
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler

from mltk import util

__all__ = [
    "DualOptimizer",
    "DiffLessThan",
    "Newton",
    "jacobian",
    "compute_gradient",
    "compute_hessian",
    "run_optim_step",
    "stop_after_n_iter",
    "optimize",
    "newton_solve"
]

def _visit_params(wrt) -> Iterable[th.Tensor]:
    if isinstance(wrt, Optimizer):
        return (param for group in wrt.param_groups for param in group["params"])
    elif not isinstance(wrt, list):
        return [wrt]
    else:
        return wrt

def jacobian(objectives: th.Tensor, wrt: th.Tensor, n_batch_axes: Optional[int] = None):
    objectives_dims = objectives.dim()
    # By default, number of batch axes is number of objectives dimensions
    if n_batch_axes is None:
        n_batch_axes = objectives_dims
    # Check number of batch axes
    elif n_batch_axes<0 or n_batch_axes>objectives_dims:
        raise ValueError("Number of batch axes must be positive and not greater than {}".format(
            objectives_dims
        ))
    # Flatten objectives and obtain its shape
    objectives = objectives.unsqueeze(-1).flatten(start_dim=n_batch_axes)
    objectives_shape = objectives.shape
    # Compute number of variables of input
    wrt_shape = wrt.shape
    # Check if shape of objective and input tensors match
    if objectives_shape[:-1]!=wrt_shape[:n_batch_axes]:
        expected_wrt_shape_repr = "({}, *)".format(", ".join(objectives_shape[:-1]))
        raise ValueError("Expect input tensor of shape {}, got {}".format(
            expected_wrt_shape_repr, wrt_shape
        ))
    # Number of objectives and input variables
    n_objectives = objectives_shape[-1]
    n_wrt_vars = wrt_shape[n_batch_axes:].numel()
    # Jacobian tensor
    jacobian = th.empty(
        (*objectives_shape, n_wrt_vars), dtype=objectives.dtype, device=objectives.device
    )
    # Store gradient of input
    wrt_grad = wrt.grad
    # Compute Jacobian for each objective
    for i in range(n_objectives):
        wrt.grad = None
        objectives[..., i].sum().backward(retain_graph=True)
        # Flatten and store Jacobian for objective
        jacobian[..., i, :] = wrt.grad.unsqueeze(-1).flatten(start_dim=n_batch_axes)
    # Restore gradient of input
    wrt.grad = wrt_grad
    return jacobian

def compute_gradient(objectives: th.Tensor, wrt, **kwargs):
    # Compute gradient in parallel for all objectives
    objectives.sum().backward(**kwargs)

def compute_hessian(objectives: th.Tensor, wrt):
    # Set number of batch axes for parameters
    n_batch_axes = objectives.dim()
    # Compute gradient for all parameters
    compute_gradient(objectives, wrt, create_graph=True)
    # Compute and set Hessian for each parameter
    for param in _visit_params(wrt):
        param.hessian = jacobian(param.grad, param, n_batch_axes)

def newton_solve(fs, init_params, stop_cond, device: Union[th.device, str] = "cpu",
    mask=None):
    # Number of functions and initial parameter values
    n_equations = len(fs)
    n_init_params = len(init_params) \
        if isinstance(init_params, (list, tuple)) \
        else np.shape(init_params)[-1]
    # Check number of initial parameter values
    if n_equations!=n_init_params:
        raise ValueError("Expect {} initial parameter values, got {}".format(
            n_equations, n_init_params
        ))
    # Initialize parameters
    if isinstance(init_params, (list, tuple)):
        params = th.stack(init_params, dim=-1).to(device)
        params.requires_grad_()
    else:
        params = th.tensor(init_params, requires_grad=True, device=device)
    # Initialize complete flags
    batch_shape = params.shape[:-1]
    if mask is None:
        complete_flags = th.zeros(batch_shape, dtype=th.bool, device=device)
    else:
        # Check shape of the mask
        mask_shape = np.shape(mask)
        if mask_shape!=batch_shape:
            raise ValueError("Expect mask of shape {}, got {}".format(
                batch_shape, mask_shape
            ))
        # Create complete flags from mask
        complete_flags = th.tensor(mask, dtype=th.bool, device=device)
    # Zero scalar tensor
    zero_scalar = th.tensor(0, dtype=params.dtype, device=device)
    # Solving loop
    for i in count():
        # Compute objectives and its Jacobians
        split_params = [params[..., i] for i in range(n_equations)]
        objectives = th.stack([f(*split_params) for f in fs], dim=-1)
        jacobians = jacobian(objectives, params, n_batch_axes=objectives.dim()-1)
        # Compute parameter decrements
        objectives_expanded = objectives.detach().unsqueeze(-1)
        decrements = util.inverse(jacobians, 0).matmul(objectives_expanded)
        # Apply decrements to pending problems only
        decrements = th.where(
            complete_flags.unsqueeze(-1), zero_scalar, decrements.squeeze(-1)
        )
        params.data -= 0.1*decrements
        # Check stop condition
        complete_flags |= stop_cond(
            iteration=i, params=params, objectives=objectives
        )
        if th.all(complete_flags):
            break
        # Release computation graph
        objectives.detach_()
        params.grad = None
    # Detach and convert parameters
    params.detach_()
    if isinstance(init_params, (list, tuple)):
        params = [params[..., i] for i in range(n_equations)]
    # Return parameters
    return params

class Newton(Optimizer):
    def __init__(self, params, lr: float = 1.):
        # Invalid learning rate for Newton method
        if lr<=0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # Initialize with default parameters
        super(Newton, self).__init__(params, defaults={
            "lr": lr
        })
    def zero_grad(self):
        # Visit each parameter
        for param in _visit_params(self):
            # Detach and clear Hessian
            hessian = getattr(param, "hessian", None)
            if hessian is not None:
                hessian.detach_()
                hessian.zero_()
            # Detach and clear gradient
            grad = param.grad
            if grad is not None:
                grad.detach_()
                grad.zero_()
    def step(self, closure=None):
        """ Perform a single optimization step with Newton method. """
        # Visit each parameter group
        for group in self.param_groups:
            lr = group["lr"]
            # Visit each parameter
            for param in group["params"]:
                grad = param.grad
                hessian = getattr(param, "hessian", None)
                # Gradient or Hessian missing
                if grad is None or hessian is None:
                    continue
                # Number of batch axes
                n_batch_axes = hessian.dim()-2
                # Reshape gradient for matrix multiplication
                grad = grad.data.unsqueeze(-1).flatten(start_dim=n_batch_axes).unsqueeze(-1)
                # Compute and reshape decrement
                decrement = util.inverse(hessian, 0).matmul(grad).reshape(param.shape)
                # Apply decrement to parameter
                param.data.add_(-lr, decrement)

def run_optim_step(objectives: th.Tensor, optimizer: Optimizer,
    backward_func=compute_gradient, lr_sched: Optional[_LRScheduler] = None):
    """ Compute gradient and perform single gradient descent step. """
    optimizer.zero_grad()
    # Invoke backward function
    backward_func(objectives, optimizer)
    # Step optimizer and zero gradient
    optimizer.step()
    # Step LR scheduler
    if lr_sched is not None:
        lr_sched.step()

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

def optimize(f, init_params, stop_cond, optimizer_factory: Optimizer = SGD,
    lr: Optional[float] = None, lr_sched_factory: Optional[_LRScheduler] = None,
    optimizer_params: dict = {}, lr_sched_params: dict = {},
    device: Union[th.device, str] = "cpu", backward_func=compute_gradient, mask=None):
    # Initialize parameters and attach backward hooks
    params = [th.tensor(param, requires_grad=True, device=device) for param in init_params]
    # Add learning rate to optimizer parameters
    if lr is not None:
        optimizer_params["lr"] = lr
    # Create optimizer and learning rate scheduler
    optimizer = optimizer_factory(params, **optimizer_params)
    lr_sched = None if lr_sched_factory is None else lr_sched_factory(
        optimizer, **lr_sched_params
    )
    # Complete flags and zero scalar tensor
    complete_flags = None if mask is None else th.tensor(mask, dtype=th.bool, device=device)
    zero_scalar = th.tensor(0, dtype=params[0].dtype, device=device)
    # Optimization loop
    for i in count():
        # Compute objectives
        objectives = f(*params)
        # Initialize complete flags if no mask is given
        if complete_flags is None:
            complete_flags = th.zeros_like(objectives, dtype=th.bool)
        # Check shape of return objectives
        complete_flags_shape = complete_flags.shape
        objectives_shape = objectives.shape
        if objectives_shape!=complete_flags_shape:
            raise ValueError("Expect objectives tensor of shape {}, got {}".format(
                complete_flags_shape, objectives_shape
            ))
        # Perform single optimization step
        run_optim_step(
            objectives=th.where(complete_flags, zero_scalar, objectives),
            optimizer=optimizer,
            lr_sched=lr_sched,
            backward_func=backward_func
        )
        # Check stop condition
        complete_flags |= stop_cond(
            iteration=i, params=params, objectives=objectives
        )
        if th.all(complete_flags):
            break
    # Detach parameters and comute objectives
    params = [param.detach() for param in params]
    objectives = f(*params)
    # Return optimized parameters and function value
    return params, f(*params)

class DualOptimizer(object):
    def __init__(self, params_backward_func=compute_gradient,
        multipliers_backward_func=compute_gradient,
        params_optimizer_factory: Optimizer = SGD,
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
