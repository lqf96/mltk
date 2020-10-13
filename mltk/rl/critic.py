from typing import Optional, SupportsFloat, SupportsInt

import torch as th

__all__ = [
    "n_step_return",
    "lambda_return"
]

def n_step_return(rewards: th.Tensor, values_last: th.Tensor, discounts: th.Tensor,
    n_steps: Optional[SupportsInt] = None) -> th.Tensor:
    n_steps = len(rewards) if n_steps is None else int(n_steps)
    
    discounts_acc = 1.
    returns = 0.
    # Accumulate rewards for each step
    for _, rewards_step, discounts_step in zip(range(n_steps), rewards, discounts):
        returns += discounts_acc*rewards_step
        discounts_acc *= discounts_step
    # Values for the last step
    returns += discounts_acc*values_last
    
    return returns

def lambda_return(rewards: th.Tensor, values_next: th.Tensor, discounts_next: th.Tensor,
    lambda_: SupportsFloat, n_steps: Optional[SupportsInt] = None) -> th.Tensor:
    lambda_ = float(lambda_)
    n_steps = len(rewards) if n_steps is None else int(n_steps)

    discounts_acc = 1.
    rewards_acc = 0.
    lambda_acc = 1.
    returns = 0.

    for i in range(n_steps):
        rewards_acc += discounts_acc*rewards[i]
        discounts_acc *= discounts_next[i]
        returns_step = rewards_acc+discounts_acc*values_next[i]
        
        weight_step = lambda_acc if i==n_steps-1 else (1-lambda_)*lambda_acc
        returns += weight_step*returns_step
        lambda_acc *= lambda_

    return returns
