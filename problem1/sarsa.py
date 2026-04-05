import numpy as np


def _sample_step(env, state_idx, action):
    """Sample (next_state_idx, reward) from the stochastic environment."""
    N        = env['N']
    next_idx = np.random.choice(N, p=env['M'][action][state_idx])
    reward   = env['R_full'][state_idx, action, next_idx]
    return next_idx, reward


def _epsilon_greedy(Q, state_idx, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state_idx])


def _greedy_path_found(Q, env, max_steps):
    """
    Check whether the greedy policy derived from Q finds a path
    from start to goal within max_steps (stochastic simulation).
    """
    s2i   = env['state_to_idx']
    start = s2i[env['start_state']]
    goal  = s2i[env['goal_state']]
    curr  = start
    for _ in range(max_steps):
        if curr == goal:
            return True
        action = np.argmax(Q[curr])
        curr, _ = _sample_step(env, curr, action)
    return curr == goal



# SARSA
def sarsa(env, n_episodes=1000, max_steps=1000,
          alpha=0.25, gamma=0.96, epsilon=0.1):

    N         = env['N']
    s2i       = env['state_to_idx']
    start_idx = s2i[env['start_state']]
    goal_idx  = s2i[env['goal_state']]

    Q               = np.zeros((N, 4))
    episode_rewards = np.zeros(n_episodes)
    convergence_ep  = n_episodes

    for ep in range(n_episodes):
        state        = start_idx
        action       = _epsilon_greedy(Q, state, epsilon)  # choose a from s
        total_reward = 0.0

        for _ in range(max_steps):
            if state == goal_idx:
                break

            next_state, reward = _sample_step(env, state, action)
            next_action        = _epsilon_greedy(Q, next_state, epsilon)

            # SARSA update: on-policy (uses next action actually taken)
            Q[state, action] += alpha * (      
                reward + gamma * Q[next_state, next_action] - Q[state, action]  # Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
            )

            total_reward += reward
            state  = next_state
            action = next_action

        episode_rewards[ep] = total_reward

        # Track first episode where greedy policy reaches goal
        if convergence_ep == n_episodes:
            if _greedy_path_found(Q, env, max_steps):
                convergence_ep = ep + 1

    policy = np.argmax(Q, axis=1)
    return Q, policy, episode_rewards, convergence_ep
