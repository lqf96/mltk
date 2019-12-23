from typing import Any, Union, Tuple, NamedTuple
from mltk.typing import StrDict

import os, traceback
from itertools import count, repeat, product
from collections.abc import Sequence
from types import SimpleNamespace

import torch as th
from torch.multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events

import mltk.util as mu
from mltk.metrics import MetricsPrinter, MetricsWriter
from .agent import RLAgent

__all__ = [
    "EPISODE_COMPLETED",
    "Step",
    "Transition",
    "ExecConfig",
    "ExecResult",
    "make_rl_engine",
    "run_rl_engine",
    "search_evaluate"
]

## Alias for episode completed event
EPISODE_COMPLETED = Events.EPOCH_COMPLETED

class Step(NamedTuple):
    ## Episode count
    episode: int
    ## Total iteration count
    iteration: int
    ## Episode iteration count
    episode_iteration: int

class Transition(NamedTuple):
    ## Previous observation of the environment
    observation: Union[th.Tensor, Tuple[th.Tensor]]
    ## Action of the agent(s)
    action: Union[th.Tensor, Tuple[th.Tensor]]
    ## Reward of the agent(s)
    reward: th.Tensor
    ## Next observation of the environment
    next_observation: Union[th.Tensor, Tuple[th.Tensor]]
    ## Current episode is done or not
    done: bool
    ## Extra information from the environment
    extra: StrDict

class ExecConfig(NamedTuple):
    ## Training mode
    training: bool = False
    ## Joint training
    joint_training: bool = False
    ## Render epoch interval
    render_interval: int = 0
    ## Render frame duration
    render_frame_duration: float = 0.5

class ExecResult(NamedTuple):
    ## All agents
    agents: list
    ## Current transition
    transition: Transition

async def _run_rl_engine(engine: Engine, _):
    """ Execute or train agent(s) in an environment. """
    engine_state = engine.state
    # Environment, agents and configuration
    env = engine_state.env
    agent = engine_state.agent
    config = engine_state.config

    # Step information
    step = engine_state.step = Step(
        episode=engine_state.epoch,
        iteration=engine_state.iteration,
        episode_iteration=engine_state.episode_iteration
    )

    # Render interval
    render_interval = config.render_interval
    # Start of new episode; reset environment
    if step.episode_iteration==1:
        observation = env.reset()
        # Render initial state
        if render_interval>0 and step.episode%render_interval==0:
            # TODO: Render asynchronously here later
            env.render()
    # Update current observation
    else:
        observation = engine_state.transition.next_observation

    # Act to current observation
    training_step = step if config.training else None
    action = agent.act(observation, training_step=training_step)
    # Run one step through the episode
    next_observation, reward, done, extra = env.step(action)

    # Render environment
    if render_interval>0 and step.episode%render_interval==0:
        await mu.delay(config.render_frame_duration)
        env.render()
        # Episode is done
        if done:
            await mu.delay(config.render_frame_duration)

    # Transition information
    transition = engine_state.transition = Transition(
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
        extra=extra
    )

    # Fit and update agent(s) during training
    if config.training:
        agent.fit(step=step, transition=transition)
        agent.update(step=step)

    # Episode ended
    if done:
        engine_state.episode_iteration = 1
        # Terminate episode
        engine.terminate_epoch()
    else:
        engine_state.episode_iteration += 1

    # Return execution result
    return ExecResult(
        agent=agent,
        transition=transition
    )

def make_rl_engine(env, agent: RLAgent, exec_config: ExecConfig = ExecConfig()):
    # Create RL engine
    engine = Engine(_run_rl_engine)

    # Initialize engine state
    @engine.on(Events.STARTED)
    def init_engine_state(engine):
        engine_state = engine.state
        # Set environment, agent and execution configuration
        engine_state.env = env
        engine_state.agent = agent
        engine_state.config = exec_config
        # Set episodic iteration count
        engine_state.episode_iteration = 1

    # Stop environment rendering
    @engine.on(Events.COMPLETED)
    def stop_env_rendering(engine):
        engine_state = engine.state
        # Stop environment rendering
        if engine_state.config.render_interval is not None:
            # TODO: Close engine asynchronously
            engine_state.env.close()

    # Return engine
    return engine

def run_rl_engine(engine: Engine, *, n_episodes: Optional[int] = None, \
    n_iterations: Optional[int] = None):
    # Episodic environment
    if n_episodes!=None:
        return engine.run(count(), max_epochs=n_episodes)
    # Infinite environment
    elif n_iterations!=None:
        return engine.run(range(n_iterations))
    # No limit is provided
    else:
        raise ValueError("Maximum number of episodes or iterations required")

# TODO: Replace this with mltk.tune in the future
def search_evaluate(env, agent_factory, search_settings_iter, fixed_settings={},
    rand=th.default_generator, exec_config=ExecConfig(), metrics_factory=None,
    n_episodes_train=None, n_iterations_train=None, n_episodes_eval=None, n_iterations_eval=None,
    report_interval=None, report_event=EPISODE_COMPLETED, writer_path="/",
    n_workers=os.cpu_count()):
    # Number of parameter combinations
    n_combs = len(search_settings_iter)
    # Parameters evaluator
    def params_evaluator(index_params):
        i, params = index_params

        # Create agent for execution
        agent = agent_factory(
            env=env,
            rand=rand,
            **fixed_settings,
            **params
        )
        # Metrics printer writer
        metrics_printer = MetricsPrinter(report_interval, report_event)
        # Metrics writer for training engine
        writer_train = SummaryWriter(os.path.join(writer_path, "train/{i}"))
        metrics_writer_train = MetricsWriter(writer_train)

        # Make RL engine for training
        engine_train = make_rl_engine(env, agent, exec_config._replace(training=True))
        # Attach metrics printer and writer to engine
        metrics_printer.attach(engine_train)
        metrics_writer_train.attach(engine_train)

        # Run engine for training
        try:
            print("[{}/{}] Training begins with settings {} and fixed settings {}".format(
                i+1, n_combs, params, fixed_settings
            ))
            run_rl_engine(
                engine_train, n_episodes=n_episodes_train, n_iterations=n_iterations_train
            )
            print("[{}/{}] Training completed".format(
                i+1, n_combs
            ))
        # Halt evaluation if exception occurs
        except Exception as e:
            traceback.print_exc()
            return None

        # Make RL engine for evaluation
        engine_eval = make_rl_engine(env, agent, exec_config._replace(training=False))
        # Metrics writer for evaluation engine
        writer_eval = SummaryWriter(os.path.join(writer_path, "test/{i}"))
        metrics_writer_eval = MetricsWriter(writer_eval)
        # Attach metrics printer to engine
        metrics_printer.attach(engine_eval)
        metrics_writer_eval.attach(engine_eval)

        # Run engine for evaluation
        try:
            print("[{}/{}] Evaluation begins with settings {} and fixed settings {}".format(
                i+1, n_combs, params, fixed_settings
            ))
            run_rl_engine(
                engine_eval, n_episodes=n_episodes_eval, n_iterations=n_iterations_eval
            )
            print("[{}/{}] Evaluation completed".format(
                i+1, n_combs
            ))
        # Halt evaluation if exception occurs
        except Exception as e:
            traceback.print_exc()
            return None

        # Return parameters for success
        return params

    # Create process pool for execution
    pool = Pool(n_workers)
    # Build and return evaluation iterator
    return pool.imap(params_evaluator, enumerate(search_settings_iter))
