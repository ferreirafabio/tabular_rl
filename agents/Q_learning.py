"""
Taken from https://github.com/automl/TabularTempoRL/
"""

import yaml
import numpy as np
import random
from collections import defaultdict
from typing import Optional

from envs.Grid import GridCore
from utils import get_env


def q_learning(
        environment: GridCore,
        num_episodes: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True):
    """
    Vanilla tabular Q-learning algorithm
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []
    test_rewards = []
    test_lens = []
    train_steps_list = []
    test_steps_list = []

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts, num_episodes, epsilon_decay)
    for i_episode in range(num_episodes + 1):
        # print('#' * 100)
        epsilon = epsilon_schedule[min(i_episode, num_episodes - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        while True:  # roll out episode
            policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            cummulative_reward += policy_reward
            episode_length += 1

            Q[policy_state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha)

            if policy_done:
                break
            policy_state = s_
        rewards.append(cummulative_reward)
        lens.append(episode_length)
        train_steps_list.append(environment.total_steps)

        # evaluation with greedy policy
        test_steps = 0
        if i_episode % eval_every == 0:
            policy_state = environment.reset()
            episode_length, cummulative_reward = 0, 0
            if render_eval:
                environment.render()
            while True:  # roll out episode
                policy_action = np.random.choice(np.flatnonzero(Q[policy_state] == Q[policy_state].max()))
                environment.total_steps -= 1  # don't count evaluation steps
                s_, policy_reward, policy_done, _ = environment.step(policy_action)
                test_steps += 1
                if render_eval:
                    environment.render()
                s_ = s_
                cummulative_reward += policy_reward
                episode_length += 1
                if policy_done:
                    break
                policy_state = s_
            test_rewards.append(cummulative_reward)
            test_lens.append(episode_length)
            test_steps_list.append(test_steps)
            print('Done %4d/%4d episodes' % (i_episode, num_episodes))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list)


def double_q_learning(
        environment: GridCore,
        num_episodes: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True):
    """
    Double tabular Q-learning algorithm following
    Algorithm 1 from https://papers.nips.cc/paper/3964-double-q-learning.pdf
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q_a = defaultdict(lambda: np.zeros(environment.action_space.n))
    Q_b = defaultdict(lambda: np.zeros(environment.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []
    test_rewards = []
    test_lens = []
    train_steps_list = []
    test_steps_list = []

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts, num_episodes, epsilon_decay)
    for i_episode in range(num_episodes + 1):
        # print('#' * 100)
        epsilon = epsilon_schedule[min(i_episode, num_episodes - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q_a, epsilon, environment.action_space.n, Q_b=Q_b)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        while True:  # roll out episode
            policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            cummulative_reward += policy_reward
            episode_length += 1

            if np.random.random() < 0.5:
                Q_a[policy_state][policy_action] = td_update(Q_a, policy_state, policy_action,
                                                             policy_reward, s_, discount_factor, alpha, Q_b)
            else:
                Q_b[policy_state][policy_action] = td_update(Q_b, policy_state, policy_action,
                                                             policy_reward, s_, discount_factor, alpha, Q_a)

            if policy_done:
                break
            policy_state = s_
        rewards.append(cummulative_reward)
        lens.append(episode_length)
        train_steps_list.append(environment.total_steps)

        # evaluation with greedy policy
        test_steps = 0
        if i_episode % eval_every == 0:
            policy_state = environment.reset()
            episode_length, cummulative_reward = 0, 0
            if render_eval:
                environment.render()
            while True:  # roll out episode
                policy_action = np.random.choice(np.flatnonzero(Q_a[policy_state] == Q_a[policy_state].max()))
                environment.total_steps -= 1  # don't count evaluation steps
                s_, policy_reward, policy_done, _ = environment.step(policy_action)
                test_steps += 1
                if render_eval:
                    environment.render()
                s_ = s_
                cummulative_reward += policy_reward
                episode_length += 1
                if policy_done:
                    break
                policy_state = s_
            test_rewards.append(cummulative_reward)
            test_lens.append(episode_length)
            test_steps_list.append(test_steps)
            print('Done %4d/%4d episodes' % (i_episode, num_episodes))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list)


def make_epsilon_greedy_policy(Q: defaultdict, epsilon: float, nA: int, Q_b: Optional[defaultdict] = None) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    :param Q_b: optional second Q-function for double Q learning
    """

    if Q_b is None:
        def policy_fn(observation):
            policy = np.ones(nA) * epsilon / nA
            best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
                Q[observation] == Q[observation].max()
            ))
            policy[best_action] += (1 - epsilon)
            return policy
    else:
        def policy_fn(observation):
            policy = np.ones(nA) * epsilon / nA
            double_Q = np.add(Q[observation], Q_b[observation])
            best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
                double_Q == double_Q.max()
            ))
            policy[best_action] += (1 - epsilon)
            return policy

    return policy_fn


def get_decay_schedule(start_val: float, decay_start: int, num_steps: int, type_: str):
    """
    Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_steps: Total number of steps to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == 'const':
        return np.array([start_val for _ in range(num_steps)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_steps - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_steps - decay_start), endpoint=True)])
    else:
        raise NotImplementedError


def td_update(q: defaultdict, state: int, action: int, reward: float, next_state: int, gamma: float, alpha: float,
              q_b: Optional[defaultdict] = None):
    """ Simple TD update rule """

    if q_b is None:
        # TD update
        best_next_action = np.random.choice(
            np.flatnonzero(q[next_state] == q[next_state].max()))  # greedy best next (with tie-breaking)
        td_target = reward + gamma * q[next_state][best_next_action]
        td_delta = td_target - q[state][action]
        return q[state][action] + alpha * td_delta
    else:
        # Double Q-learning TD update
        best_next_action = np.random.choice(
            np.flatnonzero(q[next_state] == q[next_state].max()))  # greedy best next (with tie-breaking)
        td_target = reward + gamma * q_b[next_state][best_next_action]
        td_delta = td_target - q[state][action]
        return q[state][action] + alpha * td_delta


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
    else:
        raise NotImplementedError
