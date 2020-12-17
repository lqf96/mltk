from __future__ import annotations

from typing import Optional
from mltk.types.gym import O, A, R

import asyncio as aio
from functools import partial

import gym
from mltk.engine import Engine, Events

from .agent import RLAgentBase
from .types import RLState, Transition

__all__ = [
    "EPISODE_STARTED",
    "EPISODE_COMPLETED",
    "RLState",
    "make_rl_engines"
]

RLEngine = Engine[RLState[O, A, R]]

# Aliased event for episode begin
EPISODE_STARTED = Events.EPOCH_STARTED
# Aliased event for episode end
EPISODE_COMPLETED = Events.EPOCH_COMPLETED

async def _run_rl_engine(engine: RLEngine[O, A, R]):
    state = engine.ensure_state()

    # Agent(s), environment and 
    agent = state.agent
    env = state.env
    training = state.training

    while True:
        # Start epoch
        await engine.start_epoch()
        # Reset environment state
        obs = env.reset()
        done = False

        while True:
            # Start iteration
            await engine.start_iteration()

            # Advance agent state with observation
            agent.observe(state, obs, done)
            
            if not done:
                # Get action taken by the agent
                action = agent.act(state)
                # Advance environment state with action
                _next_obs, reward, _done, extra = env.step(action)
                # Store transition
                state.transition = Transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=_next_obs,
                    done=_done,
                    extra=extra
                )
            else:
                # Clear transition for last step in the episode
                state.transition = None

            if training:
                # Train the agent in training mode
                metrics = agent.train(state)
                if metrics:
                    state.metrics = metrics

            # End iteration
            should_terminate = await engine.end_iteration()
            # Terminate engine if condition reached
            if should_terminate:
                return
            
            # End of episode
            if done:
                break
            # Update current observation and episode done flag
            else:
                obs = state.transition.next_observation
                done = state.transition.done
        
        # End epoch
        await engine.end_epoch()

def make_rl_engines(agent: RLAgentBase[O, A, R], training: bool = False,
    eval_interval: int = 0) -> tuple[RLEngine[O, A, R], Optional[RLEngine[O, A, R]]]:
    env = agent.env
    # Main state factory and engine
    state_factory_main = partial(RLState, agent=agent, env=agent.env, training=training)
    engine_main: RLEngine = Engine(_run_rl_engine, state_factory_main)
    # Return main engine only for evaluation mode
    if not training:
        return engine_main, None

    # Evaluation state factory and engine
    state_factory_eval = partial(RLState, agent=agent, env=gym.make(env.spec.id))
    engine_eval = Engine(_run_rl_engine, state_factory_eval)

    # Evaluation background task
    task_eval: Optional[aio.Task] = None
    # Barriers for training and evaluation engine
    barrier_train: Optional[aio.Future[None]] = None
    barrier_eval: Optional[aio.Future[None]] = None

    @engine_main.on(
        Events.ITER_COMPLETED(every=eval_interval),
        gate_events=(Events.ITER_STARTED,)
    )
    def switch_to_eval(_):
        nonlocal task_eval, barrier_train, barrier_eval

        # Start evaluation engine in the background
        if task_eval is None:
            task_eval = aio.create_task(engine_eval.run_async())

        # Resolve evaluation barrier
        if barrier_eval:
            barrier_eval.set_result(None)
            barrier_eval = None
        # Create new training barrier to be blocked on
        barrier_train = aio.Future()
        return barrier_train

    @engine_eval.on(Events.EPOCH_COMPLETED, gate_events=(Events.EPOCH_STARTED,))
    def switch_to_train(_):
        nonlocal barrier_train, barrier_eval

        # Resolve training barrier
        if barrier_train:
            barrier_train.set_result(None)
            barrier_train = None
        # Create new evaluation barrier to be blocked on
        barrier_eval = aio.Future()
        return barrier_eval

    return engine_main, engine_eval
