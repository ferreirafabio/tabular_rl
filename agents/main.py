import yaml
import numpy as np
import random

from utils import get_env
from agents.Dyna_Q import dyna_q
from agents.Q_learning import q_learning
from agents.Double_Q_learning import double_q_learning
from agents.SARSA import sarsa, n_step_sarsa


if __name__ == '__main__':
    with open("../config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    env_name = config["env_name"]
    agent_name = config["agent_name"]

    agent_config = config["agents"][agent_name]
    env_config = config["envs"][env_name]

    eval_eps = config["eval_eps"]
    seed = config["seed"]
    no_render = config["no_render"]

    episodes = agent_config["episodes"]
    env_max_steps = agent_config["env_max_steps"]
    agent_eps_decay = agent_config["agent_eps_decay"]
    agent_eps = agent_config["agent_eps"]

    np.random.seed(seed)
    random.seed(seed)

    env = get_env(env_name=env_name, env_max_steps=env_max_steps)

    if not no_render:
        # Clear screen in ANSI terminal
        print('\033c')
        print('\x1bc')

    # todo: make q_learning a class and pass config as a parameter
    # todo: perhaps write rl algorithm super class
    # todo: parameterize td_update with "max=True" for Q_learning vs SARSA (but perhaps first implement all algorithms then refactor, find commonalities etc.)
    # todo: implement Q function as a Variable
    # implement n-step SARSA, n-step q-learning?
    if agent_name == 'q_learning':
        train_data, test_data, num_steps = q_learning(env, episodes, epsilon_decay=agent_eps_decay, epsilon=agent_eps,
                                                      discount_factor=.99, alpha=.5,
                                                      eval_every=eval_eps, render_eval=not no_render)
    elif agent_name == 'double_q_learning':
        train_data, test_data, num_steps = double_q_learning(env, episodes, epsilon_decay=agent_eps_decay,
                                                             epsilon=agent_eps, discount_factor=.99, alpha=.5,
                                                             eval_every=eval_eps, render_eval=not no_render)
    elif agent_name == 'sarsa':
        train_data, test_data, num_steps = sarsa(env, episodes, epsilon_decay=agent_eps_decay,
                                                 epsilon=agent_eps, discount_factor=.99, alpha=.5,
                                                 eval_every=eval_eps, render_eval=not no_render)
    elif agent_name == 'dyna_q':
        mem_size = agent_config['mem_size']
        model_samples = agent_config['model_samples']
        train_data, test_data, num_steps = dyna_q(env, episodes, epsilon_decay=agent_eps_decay,
                                                  epsilon=agent_eps, discount_factor=.99, alpha=.5,
                                                  eval_every=eval_eps, render_eval=not no_render,
                                                  memory_size=mem_size, sample_n_steps_from_model=model_samples)
    elif agent_name == 'n_step_sarsa':
        n = agent_config['n']
        train_data, test_data, num_steps = n_step_sarsa(env, episodes, epsilon_decay=agent_eps_decay,
                                                 epsilon=agent_eps, discount_factor=.99, alpha=.5,
                                                 eval_every=eval_eps, render_eval=not no_render, n=n)

    else:
        raise NotImplementedError