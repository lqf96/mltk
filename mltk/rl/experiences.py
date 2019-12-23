from typing import Tuple

import torch as th

__all__ = [
    "ReplayBuffer",
    "PrioritizedReplay"
]

class ReplayBuffer():
    def __init__(self, size: Optional[int] = None, rand: th.Generator = th.default_generator):
        ## Maximum size
        self.size = size
        ## Random number generator
        self.rand = rand

        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []

    def __len__(self):
        return len(self._obs)

    def add(self, transition):
        ob, action, reward, done, ob_next = transition
        if not self._obs:
            self._obs.append(ob)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        if not done:
            self._obs.append(ob_next)

    def clear(self):
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()

    def all(self, n_steps: int = 1):
        obs = th.tensor(self._obs)
        actions = th.tensor(self._actions)
        rewards = th.tensor(self._rewards)
        dones = th.tensor(self._dones)

        # TODO: Experiences
        return [
            (obs, actions, rewards, dones) \
            for i in range(n_steps+1)
        ]

    def sample(self, batch_size: int, n_steps: int = 1):
        obs = th.tensor(self._obs)
        actions = th.tensor(self._actions)
        rewards = th.tensor(self._rewards)
        dones = th.tensor(self._dones)

        indices = th.randint(len(self), (batch_size,), generator=self.rand)

        # TODO: Experiences
        return [
            (obs, actions, rewards, dones) \
            for i in range(n_steps+1)
        ]

class PrioritizedReplay(ReplayBuffer):
    def __init__(self):
        raise NotImplementedError
