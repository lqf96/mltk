from typing import Any

import torch as th

from ..agent import MARLAgent

class ActorCriticAgent(MARLAgent):
    def __init__(self, discount_factor: float, init_q_lr: float, init_exploration: float,
        init_policy_lr: float, ac_net_factory, q_lr_decay=None, exploration_decay=None,
        policy_lr_decay=None, lola: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        # [ Naive Actor-Critic ]
        ## Discount factor
        self.discount_factor = discount_factor
        ## Initial Q-value learning rate
        self.init_q_lr = init_q_lr
        ## Initial exploration ratio
        self.init_exploration = init_exploration
        ## Initial policy learning rate
        self.init_policy_lr = init_policy_lr
        ## Q-value learning rate decay
        self.q_lr_decay = q_lr_decay
        ## Exploration decay
        self.exploration_decay = exploration_decay
        ## Policy learning rate decay
        self.policy_lr_decay = policy_lr_decay

        ## Actor-critic network
        self._ac_net = ac_net_factory(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        
        # [ Learning with Opponent-Learning Awareness (LOLA) ]
        ## LOLA flag
        self.lola = lola
        ## 

    def policy(self, observations: th.Tensor):
        pass

    def fit(self):
        pass

    def update(self):
        pass
