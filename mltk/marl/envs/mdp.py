from typing import Optional, Union, Tuple

import warnings

import numpy as np
import torch as th
from torch.distributions import Categorical, Exponential
from gym import Env
from gym.spaces import Discrete, MultiDiscrete

from mltk import util as mu

__all__ = [
    "RandomMDPEnv"
]

class RandomMDPEnv(Env):
    # Reward correlation scale
    _REWARD_CORRELATION_SCALE = 10

    def __init__(self, n_states: int, n_actions: Union[int, Tuple[int]], n_agents: Optional[int] = None,
        acyclic: bool = False, reward_correlation=None, reward_perturbation=0,
        rand: th.Generator = th.default_generator):
        super().__init__()

        # All agents have same number of actions
        if mu.isscalar(n_actions):
            # Number of agents must be given
            if n_agents is None:
                raise ValueError(
                    "Number of agents must be given when number of actions is scalar"
                )
            n_actions = (n_actions,)*n_agents
        # Check size of number of actions array
        n_actions_size = len(n_actions)
        if n_actions_size!=n_agents:
            raise ValueError("Expect {} number of actions for each agent, got {}".format(
                n_agents, n_actions_size
            ))

        # Rewards of different agents have no correlation by default
        reward_correlation = th.as_tensor(reward_correlation)
        if reward_correlation is None:
            reward_correlation = self._REWARD_CORRELATION_SCALE*th.eye(n_agents)
        # Check shape of the correlation matrix
        elif reward_correlation.shape!=(n_agents, n_agents):
            raise ValueError("Rewards correlation matrix must be a {}*{} square matrix".format(
                n_agents, n_agents
            ))

        # Full shape and allowed dimensions of reward perturbation
        perturbation_full_shape = (n_states, *n_actions, n_states, n_agents)
        perturbation_allowed_dims = [0, 1, n_agents+1, n_agents+2, n_agents+3]
        # Check shape and dimensions of the reward perturbation
        reward_perturbation = th.as_tensor(reward_perturbation)
        perturbation_shape = reward_perturbation.shape
        perturbation_dims = len(perturbation_shape)
        if perturbation_dims not in perturbation_allowed_dims:
            raise ValueError("Expect reward perturbation tensor of {} dimensions, got {}".format(
                perturbation_allowed_dims, perturbation_dims
            ))
        if perturbation_full_shape[:perturbation_dims]!=perturbation_shape:
            raise ValueError("Expect reward perturbation tensor with shape {}, got {}".format(
                perturbation_full_shape[:perturbation_dims], perturbation_shape
            ))
        # Check values of reward perturbation
        if (reward_perturbation<0).any():
            raise ValueError("Values of reward perturbation must be non-negative")

        ## State space
        self.observation_space = Discrete(n_states)
        ## Joint action space
        self.action_space = MultiDiscrete(n_actions)

        ## Reward correlation matrix
        self.reward_correlation = reward_correlation
        ## Reward perturbation
        self.reward_perturbation = reward_perturbation
        ## Acyclic MDP
        self.acyclic = acyclic
        ## Random number generator
        self.rand = rand

        # Make multi-agent MDP environment
        self._make_ma_mdp()
        # Initialize environment
        self.reset()

    @property
    def _done(self):
        # Game is done if MDP is acyclic and last state is reached
        return self.acyclic and self._state==self.n_states-1

    def _make_ma_mdp(self):
        joint_action_shape = self.joint_action_shape
        n_states = self.n_states
        n_agents = len(joint_action_shape)
        rand = self.rand

        # Reward perturbation
        perturbation = mu.unsqueeze(
            self.reward_perturbation, -1, n_states+3-self.reward_perturbation.dim()
        )
        # Generate transition probability tensor
        trans_prob = th.rand(n_states, *joint_action_shape, n_states, generator=rand)
        # Acyclic (episodic) MDP
        if self.acyclic:
            states_idx, next_states_idx = th.tril_indices(n_states)
            trans_prob[states_idx, ..., next_states_idx] = 0
        # Normalize transition probability matrix
        trans_prob /= trans_prob.sum(dim=-1, keepdim=True)
        trans_prob[th.isnan(trans_prob)] = 0

        # Generate random reward (following method ensures enough variance in rewards)
        # 1) Generate rewards "core" for state, joint actions and agents
        rewards = th.randn(n_states, *joint_action_shape, 1, n_agents, generator=rand)
        # 2) Multiply "core" by scales to generate different rewards for next state
        scales_dist = Exponential(th.tensor(1.))
        with mu.use_rand(rand):
            rewards *= scales_dist.sample(
                (n_states, *joint_action_shape, n_states, n_agents)
            )
        # 3) Correlate rewards
        rewards = rewards@self.reward_correlation
        
        ## Transition probability
        self._trans_prob = trans_prob
        ## Rewards for state-joint actions
        self._rewards = rewards

    def reset(self):
        ## Current state
        self._state = state = th.tensor(0)
        # Return current state
        return state

    def step(self, actions):
        reward_perturbation = self.reward_perturbation
        n_states = self.n_states
        n_agents = len(self.action_space.nvec)
        rand = self.rand
        state = self._state

        # Validity of joint actions
        if not self.action_space.contains(np.array(actions)):
            raise ValueError("Joint actions {} is invalid".format(actions))
        # Game already done
        if self._done:
            warnings.warn("Attempting to step the environment after game is done")
            # Dummy step result
            return state, th.zeros(n_agents), True, {}
    
        # Find transition probability distribution for state-joint actions
        trans_prob_sa = self._trans_prob[state][actions]
        # Draw next state from distribution
        next_state_dist = Categorical(probs=trans_prob_sa)
        with mu.use_rand(rand):
            self._state = next_state = next_state_dist.sample()
        
        # Get reward perturbation
        perturbation_dims = reward_perturbation.dim()
        perturbation_idx = (state, *actions, next_state)
        if perturbation_dims<=len(perturbation_idx):
            perturbation = th.full(
                n_agents, reward_perturbation[perturbation_idx[:perturbation_dims]]
            )
        else:
            perturbation = reward_perturbation[perturbation_idx]
        # Sample perturbation rewards
        perturbation_rewards = rand.normal(0., perturbation)

        # Compute total rewards
        rewards = self._rewards[state][actions][next_state].clone()
        rewards += perturbation_rewards
        # Step result
        return next_state, rewards, self._done, {}
