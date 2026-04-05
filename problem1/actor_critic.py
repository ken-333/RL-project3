import numpy as np

def _sample_step(env, state_idx, action):
    """Sample (next_state_idx, reward) from the stochastic environment."""
    N        = env['N']
    next_idx = np.random.choice(N, p=env['M'][action][state_idx])
    reward   = env['R_full'][state_idx, action, next_idx]
    return next_idx, reward


def _softmax_policy(H, state_idx):
    """
    Compute π(·|s) = softmax(H[s,:]).
    Returns probability vector of shape (n_actions,).
    """
    h    = H[state_idx]
    h    = h - np.max(h)        # for numerical stability, note this doesn't change the resulting probabilities
    exp_h = np.exp(h)
    return exp_h / exp_h.sum()


def _greedy_path_found(H, env, max_steps):
    """
    Check whether the greedy policy (argmax of H) finds a path
    from start to goal within max_steps (stochastic simulation).
    """
    s2i   = env['state_to_idx']
    start = s2i[env['start_state']]
    goal  = s2i[env['goal_state']]
    curr  = start
    for _ in range(max_steps):
        if curr == goal:
            return True
        action = np.argmax(H[curr])
        curr, _ = _sample_step(env, curr, action)
    return curr == goal


# Tabular Actor-Critic
def actor_critic(env, n_episodes=1000, max_steps=1000,  #
                 alpha=0.25, beta=0.05, gamma=0.96):

    N         = env['N']
    n_actions = len(env['M'])            # number of actions (4 for maze)
    s2i       = env['state_to_idx']
    goal_idx  = s2i[env['goal_state']]

    V               = np.zeros(N)               # critic: state value
    H               = np.zeros((N, n_actions))  # actor:  action preferences
    episode_rewards = np.zeros(n_episodes)
    convergence_ep  = n_episodes

    for ep in range(n_episodes):
        state        = np.random.randint(N)   # random start state
        total_reward = 0.0
        t            = 0

        while t < max_steps:
            if state == goal_idx:
                break

            # Select action ~ π(·|s) = softmax(H[s,:])
            pi     = _softmax_policy(H, state)
            action = np.random.choice(n_actions, p=pi)

            next_state, reward = _sample_step(env, state, action)

            # TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Critic update
            V[state] += alpha * delta

            # Actor update: increase preference for good actions
            H[state, action] += beta * delta * (1.0 - pi[action])

            total_reward += reward
            state = next_state
            t    += 1

        episode_rewards[ep] = total_reward

        # Track first episode where greedy policy reaches goal
        if convergence_ep == n_episodes:
            if _greedy_path_found(H, env, max_steps):
                convergence_ep = ep + 1

    policy = np.argmax(H, axis=1)
    return V, H, policy, episode_rewards, convergence_ep
