from typing import SupportsFloat, SupportsInt, Union

import torch as th

__all__ = [
    "n_step_return",
    "lambda_return"
]

def n_step_return(rewards: th.Tensor, values_last: th.Tensor, discount_factor: SupportsFloat,
    n_steps: Union[SupportsInt, th.Tensor, None] = None):
    discount_factor = float(discount_factor)
    # Infer number of steps from inputs
    n_steps = len(rewards) if n_steps is None else n_steps
    n_steps = th.as_tensor(n_steps)
    
    n_steps_max = n_steps.max()
    weight_step = 1.
    returns = 0.
    # Accumulate rewards for each step
    for i, rewards_step in zip(range(n_steps_max), rewards):
        # Accumulate reward for current step
        returns += (i<n_steps)*weight_step*rewards_step
        # Update weight
        weight_step *= discount_factor
    # Add values for the last step
    returns += (discount_factor**n_steps)*values_last
    
    return returns

def lambda_return(rewards: th.Tensor, values_next: th.Tensor, discount_factor: SupportsFloat,
    lambda_: SupportsFloat, n_steps: Union[SupportsInt, th.Tensor, None] = None):
    discount_factor = float(discount_factor)
    lambda_ = float(lambda_)
    # Infer number of steps from inputs
    n_steps = len(rewards) if n_steps is None else n_steps
    n_steps = th.as_tensor(n_steps)
    
    n_steps_max = n_steps.max()
    weight_values_base = (1-lambda_)*discount_factor
    weight_step = 1.
    returns = 0.
    # Accumulate N-step return for each step
    for i, rewards_step, values_next_step in zip(range(n_steps_max), rewards, values_next):
        # Weight for next values
        weight_values = (
            (i==n_steps-1)*discount_factor+
            (i<n_steps-1)*weight_values_base
        )*weight_step
        # Weight for rewards
        weight_rewards = (i<n_steps)*weight_step
        
        # Accumulate N-step return for current step
        returns += weight_rewards*rewards_step+weight_values*values_next_step
        # Update base weight
        weight_step *= discount_factor*lambda_
    
    return returns
