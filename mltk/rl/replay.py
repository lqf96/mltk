from typing import List, Optional, Tuple
from mltk.types import StrDict
from mltk.types.gym import _AbstractEnv

import torch as th
from sortedcontainers import SortedList

import mltk.util as mu
from mltk.adt import Deque, RecDeque, RecDequeSchema
from .types import Transition

__all__ = [
    "ReplayBuffer"
]

class ReplayBuffer():
    def __init__(self, env: "_AbstractEnv", capacity: int, extra_schema: RecDequeSchema = {},
        end_defaults: StrDict = {}, dtype: Optional[th.dtype] = None,
        rand: th.Generator = th.default_generator):
        self.dtype = dtype = th.get_default_dtype() if dtype is None else dtype
        self.rand = rand
        self.end_defaults = end_defaults

        obs_space = env.observation_space
        action_space = env.action_space
        # Data type for observation and action space
        obs_dtype = mu.force_float(mu.as_th_dtype(obs_space.dtype), dtype)
        action_dtype = mu.force_float(mu.as_th_dtype(action_space.dtype), dtype)

        replay_schema: RecDequeSchema = dict(
            extra_schema,
            observation=(obs_space.shape, obs_dtype),
            action=(action_space.shape, action_dtype),
            reward=dtype,
            done=th.bool
        ) # type: ignore
        # Create replay data buffer from the schema
        self.buf = buf = RecDeque.from_schema(replay_schema, max_len=capacity)

        # Number of steps in the replay buffer
        self._n_steps = 0
        # Steps of episode ends
        self._episode_end_steps = SortedList()
    
    @property
    def capacity(self) -> int:
        observations: Deque = self.buf.observation

        # Capacity of the replay buffer is always bounded
        assert observations.max_len is not None

        return observations.max_len

    def append(self, transition: Transition, **kwargs):
        buf = self.buf

        # Add information for the current step
        buf.append(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            done=False,
            **kwargs
        )
        self._n_steps += 1
        
        if transition.done:
            # Add a dummy step for end of the episode
            buf.append(
                observation=transition.next_observation,
                done=True,
                **self.end_defaults
            )
            # Record episode end
            self._episode_end_steps.add(self._n_steps)
            self._n_steps += 1
    
    def sample_seqs(self, n_seqs: int, seq_len: int) -> List[Tuple[int, int]]:
        rand = self.rand
        buf = self.buf
        episode_end_steps = self._episode_end_steps

        # Offset between number of steps and indices
        offset = self._n_steps-len(buf)
        # Maximum sequence begin index
        seq_begin_max = len(buf)-seq_len

        seq_ranges: List[Tuple[int, int]] = []
        # Sample given number of sequence indices
        while len(seq_ranges)<n_seqs:
            # Sample potential sequence range
            seq_begin_index = th.randint(0, seq_begin_max, (), generator=rand).item()
            seq_end_index = seq_begin_index+seq_len
            
            # Ignore sequence if it spans across multiple episodes
            episode_end_steps = next(episode_end_steps.irange(
                seq_begin_index+offset, seq_end_index+offset-1, inclusive=(True, False)
            ), None)
            if episode_end_steps is not None:
                continue
            # Store sequence range
            seq_ranges.append((seq_begin_index, seq_end_index))

        return seq_ranges
