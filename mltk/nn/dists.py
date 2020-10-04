from typing import SupportsFloat, Union, overload
from mltk.types import Pair

import torch as th
import torch.distributions as thd
from torch.nn import Module
import torch.nn.functional as f

import mltk.util as mu

__all__ = [
    "Categorical",
    "MultivariateNormalDiag",
    "Normal"
]

class Categorical(Module):
    __slots__ = ("_pass_logits",)

    def __init__(self, pass_logits: bool = True):
        self._pass_logits = pass_logits

    def forward(self, inputs: th.Tensor) -> thd.Categorical:
        return thd.Categorical(logits=inputs) if self._pass_logits else \
            thd.Categorical(probs=inputs)

class MultivariateNormalDiag(Module):
    __slots__ = ("epsilon",)

    def __init__(self, epsilon: SupportsFloat = 0.01):
        self.epsilon = float(epsilon)
    
    @overload
    def forward(self, inputs: th.Tensor) -> thd.LowRankMultivariateNormal: ...

    @overload
    def forward(self, inputs: Pair[th.Tensor]) -> thd.LowRankMultivariateNormal: ...

    def forward(self, inputs: Union[th.Tensor, Pair[th.Tensor]]
        ) -> thd.LowRankMultivariateNormal:
        # Separated mean and raw standard deviation
        if isinstance(inputs, tuple):
            mean, std_raw = inputs
        # Combined mean and raw standard deviation in one tensor
        else:
            combined_dims = inputs.shape[-1]
            mean_std_dims, remainder = divmod(combined_dims, 2)
            # Check shape of the combined inputs
            if remainder!=0:
                raise ValueError(
                    "expect the shape of the last dimension of combined inputs to be even, "
                    f"got {combined_dims}"
                )

            # Split combined inputs
            mean, std_raw = th.split(inputs, mean_std_dims, -1)

        return mu.MultivariateNormalDiag(
            loc=mean, scale_diag=f.softplus(std_raw)+self.epsilon
        )

class Normal(Module):
    __slots__ = ("epsilon",)

    def __init__(self, epsilon: SupportsFloat = 0.01):
        self.epsilon = float(epsilon)
    
    @overload
    def forward(self, inputs: th.Tensor) -> thd.Normal: ...

    @overload
    def forward(self, inputs: Pair[th.Tensor]) -> thd.Normal: ...

    def forward(self, inputs: Union[th.Tensor, Pair[th.Tensor]]) -> thd.Normal:
        # Separated mean and raw standard deviation
        if isinstance(inputs, tuple):
            mean, std_raw = inputs
        # Combined mean and raw standard deviation in one tensor
        else:
            # Check shape of the combined inputs
            combined_dims = inputs.shape[-1]
            if combined_dims!=2:
                raise ValueError(
                    "expect the shape of the last dimension of combined inputs to be 2, "
                    f"got {combined_dims}"
                )
            
            # Split combined inputs
            mean = inputs[..., 0]
            std_raw = inputs[..., 1]

        return thd.Normal(loc=mean, scale=f.softplus(std_raw)+self.epsilon)
