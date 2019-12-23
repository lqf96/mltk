from typing import Callable

import torch as th

import mltk.util as mu

__all__ = [
    "CriticFunc",
    "td",
    "td_lambda",
    "q_value",
    "q_max",
    "q_exp",
    "dueling",
    "dpg_value",
    "sql_value",
    "sql_value_discrete",
    "sac_value"
]

# Critic function type
CriticFunc = Callable[[th.Tensor, th.Tensor], th.Tensor]

def td(experiences, discount_factor: float, critic_act: CriticFunc, *,
    critic_target: Optional[CriticFunc] = None, n_steps: int = 1) -> th.Tensor:
    # Use behavioral critic as target critic
    if critic_target is None:
        critic_target = critic_act
    # Initialization
    discount = 1.
    tds = 0.
    dones = False

    # Add rewards for steps
    for _, _, rewards, step_dones in experiences[:-1]:
        # Normalize rewards
        rewards = (rewards-rewards.mean())/rewards.std()
        # Update TDs
        tds += discount*(~dones)*rewards
        dones |= step_dones
        discount *= discount_factor

    # Add values for end observations and actions
    obs_end, actions_end, _, _ = experiences[-1]
    with th.no_grad():
        tds += discount*dones*critic_target(obs_end, actions_end)
    # Subtract values for begin observations and actions
    obs_begin, actions_begin, _, _ = experiences[0]
    tds = tds-critic_act(obs_begin, actions_begin)

    return tds

def td_lambda(experiences, discount_factor: float, critic_act: CriticFunc,
    n_steps: int, lambda_: float, *, critic_target: Optional[CriticFunc] = None):
    # Use behavioral critic as target critic
    if critic_target is None:
        critic_target = critic_act
    # Initialization
    discount = 1.
    tds = 0.
    td_lambdas = 0.
    dones = False

    raise NotImplementedError

def q_value(q_func) -> CriticFunc:
    return lambda obs, actions: q_func(obs)[actions]

def q_max(q_func) -> CriticFunc:
    """ Make a critic function that returns the maximum Q-value for state. """
    return lambda obs, _: q_func(obs).max(dim=-1)[0]

def q_exp(q_func, policy_func) -> CriticFunc:
    """ Make a critic function that computes value from Q-values. """
    def _q_exp_wrapper(obs, _):
        q_values = q_func(obs)
        prob_actions = policy_func(obs).probs
        return prob_actions*q_values
    return _q_exp_wrapper

def dueling(v_func, a_func):
    raise NotImplementedError

def dpg_value(q_func: CriticFunc, policy_func) -> CriticFunc:
    return lambda obs, _: q_func(obs, policy_func(obs))

def sql_value(q_func: CriticFunc, t) -> CriticFunc:
    raise NotImplementedError

def sql_value_discrete(q_func, t) -> CriticFunc:
    return lambda obs, _: mu.softmax(q_func(obs), dims=-1, t=t)

def sac_value(q_func: CriticFunc, policy_func, t):
    def _sac_value_wrapper(obs, actions) -> th.Tensor:
        # Compute soft Q-values and log probability for actions
        q_values = q_func(obs, actions)
        log_prob_actions = policy_func(obs).log_prob(actions)
        # Compute soft values
        return q_values-t*log_prob_actions
    return _sac_value_wrapper
