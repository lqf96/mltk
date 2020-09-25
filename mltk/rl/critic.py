from typing import Optional

__all__ = [
    "n_step_return",
    "lambda_return"
]

def n_step_return(rewards, value_next, discount_factor):
    raise NotImplementedError

def lambda_return(rewards, values_next, discount_factor, lambda_, *, n: Optional[int] = None):
    # Infer number of steps from inputs
    if n is None:
        n = len(rewards)

    return_ = 0.
    weight_base = lambda_
    # Compute return
    for i, (reward, value_next) in enumerate(zip(rewards, values_next)):
        # Accumulate return for next value
        return_ += (weight_base*discount_factor)*value_next
        # Accumulate return for step reward
        return_ += (weight_base*(1-lambda_**(n-i))/(1-lambda_))*reward

        # Update base weight
        weight_base *= discount_factor*lambda_
    
    weight_total = lambda_*(1-lambda_**n)/(1-lambda_)
    # Normalize accumulated return
    return return_/weight_total
