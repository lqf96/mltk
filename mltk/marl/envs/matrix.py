import torch as th
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete

__all__ = [
    "MatrixGameEnv"
]

class MatrixGameEnv(Env):
    def __init__(self, matrices, reward_perturbation=0, rand: th.Generator = th.default_generator):
        super().__init__()
        
        matrices = th.as_tensor(matrices)
        reward_perturbation = th.as_tensor(reward_perturbation)
        # Check shape of transition matrix
        n_agents = matrices.dim()-1
        if matrices.shape[0]!=n_agents:
            raise ValueError("Number of matrices does not match dimensions of each matrix")

        # Check shape of reward perturbation
        if reward_perturbation.shape!=() and reward_perturbation.shape!=(n_agents,):
            raise ValueError("Reward perturbation must be either same or specified for each agent")
        # Check values of reward perturbation
        if (reward_perturbation<0).any():
            raise ValueError("Values of reward perturbation must be non-negative")

        ## State space
        self.observation_space = Discrete(1)
        ## Action space
        self.action_space = MultiDiscrete(matrices.shape[1:])

        ## Matrices of the matrix game
        self.matrices = matrices
        ## Standard deviation of reward perturbation
        self.reward_perturbation = reward_perturbation
        ## Random number generator
        self.rand = rand

    def reset(self):
        return th.tensor(0)

    def step(self, actions):
        # Check validity of joint actions
        if not self.action_space.contains(np.array(actions)):
            raise ValueError("Joint actions {} is invalid".format(actions))

        # Rewards for each agent
        rewards = self.matrices[(slice(None), *actions)].clone()
        # Add random perturbation to rewards
        reward_perturbation = self.reward_perturbation
        if (reward_perturbation!=0).all():
            rewards += th.normal(0., reward_perturbation, generator=self.rand)

        # Step result
        return th.tensor(0), rewards, True, {}
