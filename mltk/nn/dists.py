from typing import SupportsFloat, Union, overload
from mltk.types import Pair

import torch as th
import torch.distributions as thd
from torch.nn import Module
import torch.nn.functional as f
from torch.jit import is_tracing, is_scripting

import mltk.util as mu

__all__ = [
    "Binomial",
    "Categorical",
    "MultivariateNormalDiag",
    "Normal",
    "OneHotCategorical"
]

class Binomial(Module):
    __slots__ = ("total_count", "pass_logits")

    def __init__(self, total_count: int = 1, pass_logits: bool = True):
        super().__init__()
        
        self.total_count = total_count
        self.pass_logits = pass_logits

    def forward(self, inputs: th.Tensor) -> thd.Binomial:
        total_count = self.total_count

        return thd.Binomial(total_count, logits=inputs) \
            if self.pass_logits \
            else thd.Binomial(total_count, probs=inputs)

class Categorical(Module):
    __slots__ = ("pass_logits",)

    def __init__(self, pass_logits: bool = True):
        super().__init__()
        
        self.pass_logits = pass_logits

    def forward(self, inputs: th.Tensor) -> thd.Categorical:
        return thd.Categorical(logits=inputs) \
            if self.pass_logits \
            else thd.Categorical(probs=inputs)

class MultivariateNormalDiag(Module):
    __slots__ = ("epsilon",)

    def __init__(self, epsilon: SupportsFloat = 0.01):
        super().__init__()
        
        self.epsilon = float(epsilon)
    
    @overload
    def forward(self, inputs: th.Tensor) -> mu.MultivariateNormalDiag: ...

    @overload
    def forward(self, inputs: Pair[th.Tensor]) -> mu.MultivariateNormalDiag: ...

    def forward(self, inputs: Union[th.Tensor, Pair[th.Tensor]]
        ) -> mu.MultivariateNormalDiag:
        # Separated mean and raw standard deviation
        if isinstance(inputs, tuple):
            mean, std_raw = inputs
        # Combined mean and raw standard deviation in one tensor
        else:
            combined_dims = inputs.shape[-1]
            mean_std_dims = combined_dims//2
            # Check shape of the combined inputs
            if not is_scripting() and not is_tracing():
                remainder = combined_dims%2
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
        super().__init__()
        
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
            if not is_scripting() and not is_tracing():
                if combined_dims!=2:
                    raise ValueError(
                        "expect the shape of the last dimension of combined inputs to be 2, "
                        f"got {combined_dims}"
                    )
            
            # Split combined inputs
            mean = inputs[..., 0]
            std_raw = inputs[..., 1]

        return thd.Normal(loc=mean, scale=f.softplus(std_raw)+self.epsilon)

class OneHotCategorical(Module):
    __slots__ = ("pass_logits",)

    def __init__(self, pass_logits: bool = True):
        super().__init__()
        
        self.pass_logits = pass_logits

    def forward(self, inputs: th.Tensor) -> mu.OneHotCategorical:
        return mu.OneHotCategorical(logits=inputs) \
            if self.pass_logits \
            else mu.OneHotCategorical(probs=inputs)
