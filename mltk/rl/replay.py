from typing import Generic, Optional
from mltk.types.gym import Env

import torch as th
from sortedcontainers import SortedList

import mltk.util as mu
from mltk.adt import Deque, RecDeque, RecDequeSchema
from .types import Transition

__all__ = [
    "ReplayBuffer"
]

class ReplayBuffer():
    def __init__(self, env: "Env", capacity: int, extras_schema: RecDequeSchema = {},
        dtype: Optional[th.dtype] = None, rand: th.Generator = th.default_generator):
        self.dtype = dtype = th.get_default_dtype() if dtype is None else dtype
        self.rand = rand

        obs_space = env.observation_space
        action_space = env.action_space
        # Data type for observation and action space
        obs_dtype = mu.force_float(mu.as_th_dtype(obs_space.dtype), dtype)
        action_dtype = mu.force_float(mu.as_th_dtype(action_space.dtype), dtype)

        # Data schema of the replay buffer
        replay_schema: RecDequeSchema = dict(
            extras_schema,
            observation=(obs_space.shape, obs_dtype),
            action=(action_space.shape, action_dtype),
            reward=dtype,
            done=th.bool
        ) # type: ignore
        # Experience buffer
        self.buf = RecDeque.from_schema(replay_schema, max_len=capacity)

        self._steps = 0
        self._episode_begin_steps = SortedList()
    
    @property
    def capacity(self) -> int:
        observations: Deque = self.buf.observation

        # Capacity of the replay buffer is always bounded
        assert observations.max_len is not None
        return observations.max_len

    def append(self, transition: Transition, **kwargs):
        buf = self.buf

        # Add experience to buffer
        buf.append(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            done=transition.done,
            **kwargs
        )
        # Record begin of an episode
        if not buf or buf.done[-1]:
            self._episode_begin_steps.add(self._steps)
        # Update number of steps
        self._steps += 1
    
    def sample_seqs(self, n_seqs: int, seq_len: int) -> th.Tensor:
        rand = self.rand
        episode_begin_steps = self._episode_begin_steps
        # Current replay buffer size
        buf_len = len(self.buf)

        seq_begin_indices = th.empty((n_seqs,), dtype=th.int64)
        # Offset between number of steps and indices
        offset = self._steps-buf_len

        i = 0
        while i<n_seqs:
            # Sample sequence begin index
            seq_begin_index = th.randint(buf_len-seq_len, (), generator=rand).item()
            # Compute corresponding number of steps
            seq_begin_step = offset+seq_begin_index
            
            try:
                # Try to find episode begin within sequence range
                next(episode_begin_steps.irange(
                    seq_begin_step, seq_begin_step+seq_len, inclusive=(False, False)
                ))
                # Episode begin found; resample sequence begin index
                continue
            except StopIteration:
                # Store sequence begin index
                seq_begin_indices[i] = seq_begin_index
                i += 1

        return seq_begin_indices
