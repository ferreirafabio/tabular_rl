import numpy as np

from collections import defaultdict
from envs.Grid import GridCore
from agents.rl_helpers import get_decay_schedule, make_epsilon_greedy_policy, td_update


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
                                                             policy_reward, s_, discount_factor, alpha, policy_done,
                                                             Q_b)
            else:
                Q_b[policy_state][policy_action] = td_update(Q_b, policy_state, policy_action,
                                                             policy_reward, s_, discount_factor, alpha, policy_done,
                                                             Q_a)

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