from typing import Any, Union, Optional, Collection, Tuple
from abc import ABC, abstractmethod

import torch as th
from torch.distributions import Distribution
from gym import Env
from gym.spaces import MultiDiscrete, Tuple as TupleSpace

from mltk import util
from mltk.rl import RLAgent, Step, Transition

__all__ = [
    "MARLAlgorithm",
    "AgentsGroup",
    "MARLAgent",
    "n_agents",
    "joint_action_shape",
    "reduction_dims"
]

def n_agents(env: Env) -> int:
    """ Number of agents in the game. """
    action_space = env.action_space

    # Tuple action space
    if isinstance(action_space, TupleSpace):
        return len(action_space.spaces)
    # Multi-discrete action space
    elif isinstance(action_space, MultiDiscrete):
        return len(action_space.nvec)
    # Unsuuported action space
    else:
        raise TypeError(
            f"Action space {action_space} has unknown space type {action_space.__class__}"
        )

def joint_action_shape(env: Env) -> Tuple[int]:
    """ Shape of discrete joint action space. """
    action_space = env.action_space

    # Tuple action space
    if isinstance(action_space, TupleSpace):
        return tuple((space.n for space in action_space.spaces))
    # Multi-discrete action space
    elif isinstance(action_space, MultiDiscrete):
        return tuple(action_space.nvec)
    # Unsupported action space
    else:
        raise TypeError(
            f"Action space {action_space} has unknown space type {action_space.__class__}"
        )

def reduction_dims(n_agents: int, exclude_agent: Optional[int] = None):
    # Reduction dimensions
    dims = list(range(-n_agents, 0))
    # Remove agent from dimensions
    if exclude_agent!=None:
        dims.remove(exclude_agent-n_agents)

    return tuple(dims)

# TODO: Consider having an Abstract RL agent and inheriting from it
class MARLAlgorithm(RLAgent):
    @abstractmethod
    def policy(self, observations: Union[th.Tensor], training_step: Optional[Step] = None) \
        -> Tuple[Union[Distribution, th.Tensor]]:
        raise NotImplementedError

    n_agents = property(lambda self: n_agents(self.env))

    joint_action_shape = property(lambda self: joint_action_shape(self.env))

class AgentsGroup(MARLAlgorithm):
    def __init__(self, agents: Collection["MARLAgent"], joint_training: bool = False):
        ## All agents
        self.agents = agents
        ## Joint training flag
        self.joint_training = joint_training

        # Override each agent's setting
        for agent in agents:
            agent.joint_training = joint_training

    def policy(self, obs, **kwargs):
        raise NotImplementedError

    def act(self, obs, **kwargs):
        return tuple((agent.act(obs, **kwargs) for agent in self.agents))

    def update(self, step: Step, transition: Transition):
        agents = self.agents

        for agent in agents:
            all_agents = agents if self.joint_training else None
            agent.update(step=step, transition=transition, all_agents=all_agents)

class MARLAgent(RLAgent):
    def __init__(self, index: int, joint_training: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        ## Index of current agent
        self.index = index
        ## Whether all agents are trained jointly or not
        self.joint_training = joint_training

    def update(self, step: Step, transition: Transition,
        all_agents: Optional[Tuple["MARLAgent"]] = None):
        raise NotImplementedError

    n_agents = property(lambda self: n_agents(self.env))

    joint_action_shape = property(lambda self: joint_action_shape(self.env))

    @property
    def n_actions(self) -> int:
        action_space = self.action_space
        self_idx = self.index

        # Tuple action space
        if isinstance(action_space, TupleSpace):
            return action_space.spaces[self_idx].n
        # Multi-discrete action space
        elif isinstance(action_space, MultiDiscrete):
            return action_space.nvec[self_idx]
        # Unsupported action space
        else:
            raise TypeError("Action space {} has unknown discrete space type {}".format(
                action_space, action_space.__class__
            ))

    def reduction_dims(self, exclude_agent: Optional[int] = None, exclude_self: bool = False):
        # Exclude dimension of current agent
        if exclude_agent is None and exclude_self:
            exclude_agent = self.index

        return reduction_dims(self.n_agents, exclude_agent)
