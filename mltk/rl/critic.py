from typing import Iterable, Optional, SupportsFloat, SupportsInt, Union

import torch as th

__all__ = [
    "n_step_return",
    "lambda_return"
]

def n_step_return(rewards: Iterable[th.Tensor], values_last: th.Tensor,
    discounts: Union[SupportsFloat, th.Tensor], n_steps: Optional[SupportsInt] = None
    ) -> th.Tensor:
    n_steps = len(rewards) if n_steps is None else int(n_steps)
    if not isinstance(discounts, th.Tensor):
        discounts = rewards.new_tensor(discounts).expand_as(rewards)
    
    discounts_acc = 1.
    returns = 0.
    # Accumulate rewards for each step
    for _, rewards_step, discounts_step in zip(range(n_steps), rewards, discounts):
        returns += discounts_acc*rewards_step
        discounts_acc *= discounts_step
    # Values for the last step
    returns += discounts_acc*values_last
    
    return returns

def lambda_return(rewards: Iterable[th.Tensor], values_next: th.Tensor,
    discounts_next: Union[SupportsFloat, th.Tensor], lambda_: SupportsFloat,
    n_steps: Optional[SupportsInt] = None) -> th.Tensor:
    lambda_ = float(lambda_)
    n_steps = len(rewards) if n_steps is None else int(n_steps)
    if not isinstance(discounts_next, th.Tensor):
        discounts_next = rewards.new_tensor(discounts_next).expand_as(rewards)

    discounts_acc = 1.
    rewards_acc = 0.
    lambda_acc = 1.
    returns = 0.

    for i in range(n_steps):
        rewards_acc = rewards_acc+discounts_acc*rewards[i]
        discounts_acc = discounts_acc*discounts_next[i]
        returns_step = discounts_acc*values_next[i]
        
        weight_step = lambda_acc if i==n_steps-1 else (1-lambda_)*lambda_acc
        returns = returns+weight_step*returns_step
        lambda_acc *= lambda_

    return returns
