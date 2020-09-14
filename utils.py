

def get_env(env_name, env_max_steps):
    d = None

    if env_name.startswith('lava'):
        from envs.Grid import Bridge6x10Env, Pit6x10Env, ZigZag6x10, ZigZag6x10H

        perc = env_name.endswith('perc')
        ng = env_name.endswith('ng')
        if env_name.startswith('lava2'):
            d = Bridge6x10Env(max_steps=env_max_steps, percentage_reward=perc, no_goal_rew=ng)
        elif env_name.startswith('lava3'):
            d = ZigZag6x10(max_steps=env_max_steps, percentage_reward=perc, no_goal_rew=ng, goal=(5, 9))
        elif env_name.startswith('lava4'):
            d = ZigZag6x10H(max_steps=env_max_steps, percentage_reward=perc, no_goal_rew=ng, goal=(5, 9))
        else:
            d = Pit6x10Env(max_steps=env_max_steps, percentage_reward=perc, no_goal_rew=ng)

        return d

    else:
        raise NotImplemented("Environment not supported")


