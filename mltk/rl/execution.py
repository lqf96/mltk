from typing import Generic, cast
from mltk.types import StrDict
from mltk.types.gym import O, A, R, _AbstractEnv

from mltk.engine import Engine, Events, State

from .agent import _AbstractRLAgent
from .types import Step, Transition

__all__ = [
    "EPISODE_STARTED",
    "EPISODE_COMPLETED",
    "RLState",
    "make_rl_engine"
]

# Aliased event for episode begin
EPISODE_STARTED = Events.EPOCH_STARTED
# Aliased event for episode end
EPISODE_COMPLETED = Events.EPOCH_COMPLETED

class RLState(State, Generic[O, A, R]):
    agent: _AbstractRLAgent[O, A, R]
    env: _AbstractEnv[O, A, R]
    training: bool
    transition: Transition
    extra: StrDict

async def _run_rl_engine(engine: Engine):
    """ Execute or train agent(s) in an environment. """
    engine_state = cast(RLState[O, A, R], engine.state)

    # Environment, agents and training flag
    env = engine_state.env
    agent = engine_state.agent
    training = engine_state.training

    async for episodes in engine.epochs():
        obs = env.reset()

        async for episode_iterations in engine.iterations():
            # Step information
            step = Step(
                episodes=episodes,
                iterations=engine_state.iterations,
                episode_iterations=episode_iterations
            )

            # Act to current observation
            action = agent.act(obs, training_step=step if training else None)
            # Run one step through the episode
            next_obs, reward, done, extra = env.step(action)

            # Transition information
            transition = engine_state.transition = Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
                extra=extra
            )

            # Update experience
            agent.update_experiences(step=step, transition=transition)
            # Update agent during training
            if training:
                extra = agent.update_policy(step=step)
                if extra:
                    engine_state.extra = extra
            
            # Update observation
            if done:
                break
            else:
                obs = next_obs
        
        # Terminate current iteration
        await engine.emit(Events.ITER_COMPLETED)
    
    # Terminate current epoch
    await engine.emit(Events.EPOCH_COMPLETED)

def make_rl_engine(env, agent: _AbstractRLAgent[O, A, R], training: bool = False):
    # Create RL engine
    engine = Engine(_run_rl_engine)

    @engine.on(Events.STARTED)
    def init_engine_state(engine: Engine):
        engine_state = cast(RLState[O, A, R], engine.state)

        # Set envionrment, agent and training flag
        engine_state.env = env
        engine_state.agent = agent
        engine_state.training = training
        
        # TODO: Temporary
        from collections import defaultdict
        engine_state.extra = defaultdict(lambda: 0.)

    # Return engine
    return engine
