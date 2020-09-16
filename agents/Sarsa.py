import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count

from collections import defaultdict
from envs.Grid import GridCore
from agents.rl_helpers import get_decay_schedule, make_epsilon_greedy_policy, td_update, update_eligibility_trace


def sarsa(
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
    Vanilla tabular SARSA algorithm
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
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        while True:  # roll out episode
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            a_ = np.random.choice(list(range(environment.action_space.n)), p=policy(s_))
            cummulative_reward += policy_reward
            episode_length += 1

            Q[policy_state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha, policy_done,
                                                       action_=a_)

            if policy_done:
                break
            policy_state = s_
            policy_action = a_
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
            print("test rewards {}".format(test_rewards))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list)


def n_step_sarsa(
        environment: GridCore,
        num_episodes: int,
        n: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True):
    """
    Vanilla tabular n-step SARSA algorithm
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :param n: todo
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
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))

        n_step_rewards, n_step_states, n_step_actions = [], [], []

        n_step_states.append(policy_state)
        n_step_actions.append(policy_action)

        T = np.iinfo(np.int32).max
        tau = 0

        while not (tau == T-1):  # roll out episode
            if episode_length < T:
                s_, policy_reward, policy_done, _ = environment.step(policy_action)
                cummulative_reward += policy_reward

                n_step_rewards.append(policy_reward)
                n_step_states.append(s_)

                if policy_done:
                    T = episode_length + 1
                else:
                    a_ = np.random.choice(list(range(environment.action_space.n)), p=policy(s_))
                    policy_action = a_
                    n_step_actions.append(policy_action)

            tau = episode_length - n + 1
            if tau >= 0:
                G = []
                for i in range(tau+1, min(tau+n, T)):
                    G.append((discount_factor**(i-tau-1))*n_step_rewards[i])
                G = np.sum(G)
                if tau + n < T:
                    G += (discount_factor**n) * Q[n_step_states[tau+n]][n_step_actions[tau+n]]

                q_state_to_be_updated = n_step_states[tau]
                q_action_to_be_updated = n_step_actions[tau]
                Q[q_state_to_be_updated][q_action_to_be_updated] += alpha * (G - Q[q_state_to_be_updated][q_action_to_be_updated])

            episode_length += 1

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
            print("test rewards {}".format(test_rewards))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list)


def sarsa_lambda(
        environment: GridCore,
        num_episodes: int,
        lambd: float,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True,
        parallel_eligibility_updates=False):
    """
    Vanilla tabular SARSA (lambda) algorithm using eligibility traces according to Sutton's RL book (http://incompleteideas.net/book/first/ebook/node77.html)
    (beware: above reference contains a bug as the eligibility matrix needs to be reset in each episode)
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param lambd: specifies the lambda parameter
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
        # initialize eligibility matrix
        E = defaultdict(lambda: np.zeros(environment.action_space.n))
        epsilon = epsilon_schedule[min(i_episode, num_episodes - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        while True:  # roll out episode
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            a_ = np.random.choice(list(range(environment.action_space.n)), p=policy(s_))
            cummulative_reward += policy_reward
            episode_length += 1

            td_delta = (policy_reward + discount_factor * Q[s_][a_]) - Q[policy_state][policy_action]
            E[policy_state][policy_action] += 1

            if parallel_eligibility_updates:
                # parallel version is only faster for large state spaces
                pool = Pool(cpu_count())
                # starmap maintains initial order
                results = pool.starmap(update_eligibility_trace, [(s, Q[s], E[s], td_delta, alpha, discount_factor, lambd) for s in Q.keys()])
                pool.close()
                pool.join()
                for result in results:
                    Q_state, E_state, s = result
                    Q[s], E[s] = Q_state, E_state
            else:
                for s in Q.keys():
                    Q[s], E[s], s = update_eligibility_trace(s, Q_state=Q[s], E_state=E[s], td_delta=td_delta, alpha=alpha, discount_factor=discount_factor,
                                                               lambd=lambd)

            if policy_done:
                break
            policy_state = s_
            policy_action = a_

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
            print("test rewards {}".format(test_rewards))

    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list)