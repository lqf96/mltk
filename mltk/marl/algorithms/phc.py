from typing import Any, Collection

import torch as th
from torch.distributions import Categorical

from mltk.rl import Step, Transition
from mltk.optim.decay import apply_decay
from ..agent import MARLAgent

__all__ = [
    "PHCAgent"
]

class PHCAgent(MARLAgent):
    def __init__(self, discount_factor: float, init_q_lr: float, init_exploration: float,
        init_phc_lr: float, q_lr_decay=None, exploration_decay=None, phc_lr_decay=None,
        wolf_lr_ratio=None, **kwargs: Any):
        super().__init__(**kwargs)
        # Number of states and self actions
        n_states = self.n_states
        n_actions = self.n_actions

        ## [ Policy Hill Climbing ]
        ## Discount factor
        self.discount_factor = discount_factor
        ## Initial Q-value learning rate
        self.init_q_lr = init_q_lr
        ## Initial exploration ratio
        self.init_exploration = init_exploration
        ## Initial PHC learning rate
        self.init_phc_lr = init_phc_lr
        ## Q-value learning rate decay
        self.q_lr_decay = q_lr_decay
        ## Exploration decay
        self.exploration_decay = exploration_decay
        ## PHC learning rate decay
        self.phc_lr_decay = phc_lr_decay

        ## Current transition
        self._transition = None
        ## Policy
        self._policy = policy = th.full((n_states, n_actions), 1/n_actions)
        ## Q-values
        self._q_values = th.zeros((n_states, n_actions))

        ## [ WoLF ]
        ## WoLF learning rate ratio
        self.wolf_lr_ratio = wolf_lr_ratio

        ## Average policies
        self._avg_policy = th.zeros_like(policy)
        ## State count
        self._state_count = th.zeros(n_states, dtype=th.int)

    def policy(self, observations: th.Tensor, training_step: Optional[Step] = None,
        all_agents: Optional[Tuple[MARLAgent]] = None) -> Categorical:
        # Look up policy for given states
        policy = self._policy[observations]

        # Add exploration to policies
        if training_step!=None:
            exploration = apply_decay(
                self.init_exploration, self.exploration_decay,
                epoch=training_step.iteration
            )
            policy = (1-exploration)*policy+ \
                exploration*th.full_like(policy, 1/self.n_actions)

        return Categorical(probs=policy)

    def value(self, observations: th.Tensor, _use_avg_policy: bool = True) -> th.Tensor:
        policy = self._avg_policy if _use_avg_policy else self._policy
        # Policy and Q-values for observations
        policy_obs = policy[observations]
        q_obs = self._q_values[observations]

        return (policy_obs*q_obs).sum()

    def fit(self, step: Step, transition: Transition,
        all_agents: Optional[Tuple[MARLAgent]] = None):
        self_idx = self.self_idx
        q_values = self._q_values

        self._transition = transition
        # Transition information
        state, actions, rewards, next_state, done = transition.item()
        action_self = actions[self_idx]

        # Q-value learning rate
        q_lr = apply_decay(self.init_q_lr, self.q_lr_decay, epoch=step.iteration)
        # Q-value for next state
        q_next = 0 if done else q_values[next_state].max()
        # Update Q-value for current state-self action pair
        q_new = rewards[self_idx]+self.discount_factor*q_next
        q_values[state, action_self] = (1-q_lr)*q_values[state, action_self]+q_lr*q_new

    def update(self, step: Step, all_agents: Optional[Tuple[MARLAgent]] = None):
        wolf_lr_ratio = self.wolf_lr_ratio
        transition = self._transition
        # Current state
        state = transition["observation"]
        # Policy learning rate
        phc_lr = apply_decay(self.init_phc_lr, self.phc_lr_decay, epoch=step.iteration//100)

        # Q-values and policy for current state
        q_state = self._q_values[state]
        policy_state = self._policy[state]
        # Maximum Q-value action
        max_q_action = th.argmax(q_state)

        # Apply WoLF (Win or Lose Fast) principle
        if wolf_lr_ratio is not None:
            # Update state count
            self._state_count[state] += 1
            state_count = self._state_count[state]
            # Update average policy
            avg_policy_state = self._avg_policy[state]
            avg_policy_state = avg_policy_state*(state_count-1)/state_count+ \
                policy_state/state_count

            # Compute values under current and average policy
            value_current = self.value(state)
            value_average = self.value(state, _use_avg_policy=True)
            # Decrease learning rate if agent is "winning"
            if value_current>value_average:
                phc_lr /= wolf_lr_ratio

        # Constrain learning rate by maximum Q-value for current state
        phc_lr = th.min(th.tensor(phc_lr), 1-policy_state[max_q_action])
        # Compute policy change for all actions
        policy_change = th.min(policy_state, phc_lr/(self.n_actions-1))
        policy_change[max_q_action] = 0
        policy_change[max_q_action] = -th.sum(policy_change)
        # Update policy for current state
        policy_state -= policy_change
