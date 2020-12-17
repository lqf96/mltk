from __future__ import annotations

import torch as th
from torch import autograd, nn

__all__ = [
    "PowerNorm",
    "power_norm"
]

class _PowerNormFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x: th.Tensor, weight: th.Tensor, bias: th.Tensor, running_quad: th.Tensor,
        nu: th.Tensor, epsilon: float, momentum: float, training: bool) -> th.Tensor:
        # Normalize inputs by quadratic statistics
        running_quad_sqrt = running_quad.sqrt()
        x_normed = x/(running_quad_sqrt+epsilon)
        # Rescale normalized inputs
        x_rescaled = weight*x_normed+bias

        if training:
            reduce_dims = tuple(range(x.ndim-1))
            # Compute batch mean quadratic statistics
            batch_quad = x.square().mean(reduce_dims)
            # Update quadratic statistics
            running_quad.mul_(momentum).add_((1-momentum)*batch_quad)

            # Save variables for backward computation
            ctx.save_for_backward(x_normed, weight)
            # Save statistics and configurations
            ctx.running_quad_sqrt = running_quad_sqrt
            ctx.nu = nu
            ctx.momentum = momentum
            ctx.reduce_dims = reduce_dims

        return x_rescaled

    @staticmethod
    def backward(ctx, grad_outputs):
        x_normed, weight = ctx.saved_tensors
        nu = ctx.nu
        momentum = ctx.momentum
        reduce_dims = ctx.reduce_dims
        x_rescaled_grad, = grad_outputs
        
        # Compute gradient with regard to inputs
        x_normed_grad = weight*x_rescaled_grad
        x_grad = (x_normed_grad-nu*x_normed)/ctx.running_quad_sqrt
        # Compute gradient with regard to weight and bias
        weight_grad = (x_normed*x_rescaled_grad).sum(reduce_dims)
        bias_grad = x_rescaled_grad.sum(reduce_dims)

        # Compute intermediate statistics
        gamma = x_normed.square().mean(reduce_dims)
        lambda_ = (x_normed*x_normed_grad).mean(reduce_dims)
        # Update gradient statistics
        nu.mul_(1-momentum*gamma).add_(momentum*lambda_)

        return x_grad, weight_grad, bias_grad, None, None

def power_norm(x: th.Tensor, weight: th.Tensor, bias: th.Tensor, running_quad: th.Tensor,
    nu: th.Tensor, epsilon: float = 0.00001, momentum: float = 0.9, training: bool = False):
    return _PowerNormFunc.apply(x, weight, bias, running_quad, nu, epsilon, momentum, training)

class PowerNorm(nn.Module):
    running_quad: th.Tensor
    nu: th.Tensor

    def __init__(self, feat_dims: int, epsilon: float = 0.00001, momentum: float = 0.9):
        self.feat_dims = feat_dims
        self.epsilon = epsilon
        self.momentum = momentum

        self.weight = nn.Parameter(th.empty(feat_dims))
        self.bias = nn.Parameter(th.empty(feat_dims))
        self.register_buffer("running_quad", th.empty(feat_dims))
        self.register_buffer("nu", th.empty(feat_dims))

        self.reset_params()

    def reset_running_stats(self) -> None:
        self.running_quad.zero_()
        self.nu.zero_()

    def reset_params(self):
        self.reset_running_stats()
        self.weight.fill_(1)
        self.bias.zero_()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return power_norm(
            x, self.weight, self.bias, self.running_quad, self.nu,
            self.epsilon, self.momentum, self.training
        )
