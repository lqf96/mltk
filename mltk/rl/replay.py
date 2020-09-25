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
        # Add an initial dummy step to the buffer
        buf.append(
            observation=th.zeros(obs_space.shape),
            reward=0.,
            done=True,
            **end_defaults
        )

        # Number of steps in the replay buffer
        self._n_steps = 1
        # Steps of episode ends
        self._episode_end_steps = SortedList([0])
    
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
                reward=0.,
                done=True,
                **self.end_defaults
            )
            # Record episode end
            self._episode_end_steps.add(self._n_steps)
            self._n_steps += 1
    
    def sample_seqs(self, n_seqs: int, n_mem_steps: int, n_loss_steps: int
        ) -> List[Tuple[int, int, int]]:
        rand = self.rand
        buf = self.buf
        episode_end_steps = self._episode_end_steps

        # Current replay buffer size
        buf_len = len(buf)
        # Offset between number of steps and indices
        offset = self._n_steps-buf_len

        seq_indices = []
        # Sample given number of sequences; each sequence satisfies following properties:
        # 1) Sequences have no more than `n_mem_steps` of memory steps for building up the
        #    hidden state for a recurrent network.
        # 2) Sequences have no more than `n_loss_steps` of loss steps for computing RL losses.
        # 3) Sequences are bounded between neighboring episode end steps (exclusive at the
        #    beginning and inclusive at the end)
        for _ in range(n_seqs):
            # Sample sequence middle index
            seq_mid_index = th.randint(
                n_mem_steps+1, buf_len-n_loss_steps, (), generator=rand
            ).item() # type: ignore
            # Adjust index for end step of an episode
            if buf.done[seq_mid_index]:
                seq_mid_index -= 1
            # Number of steps for sequence middle
            seq_mid_steps = seq_mid_index+offset
            
            # Sequence begin index
            try:
                seq_begin_index = next(episode_end_steps.irange(
                    seq_mid_steps-n_mem_steps, seq_mid_steps, reverse=True
                ))-offset+1
            except StopIteration:
                seq_begin_index = seq_mid_index-n_mem_steps
            # Sequence end index
            try:
                seq_end_index = next(episode_end_steps.irange(
                    seq_mid_steps, seq_mid_steps+n_loss_steps
                ))-offset
            except StopIteration:
                seq_end_index = seq_mid_index+n_loss_steps
            
            # Store sequence indices
            seq_indices.append((seq_begin_index, seq_mid_index, seq_end_index))

        return seq_indices
