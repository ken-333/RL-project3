import numpy as np


# =============================================================================
# Internal helpers
# =============================================================================

def _sample_step(env, state_idx, action):
    """Sample (next_state_idx, reward) for the gene network."""
    N        = env['N']
    next_idx = np.random.choice(N, p=env['M'][action][state_idx])
    reward   = env['gene_reward'][next_idx] + env['action_cost'][action]
    return next_idx, reward


def _epsilon_greedy(Q, state_idx, N_ACTS, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTS)
    return np.argmax(Q[state_idx])


# =============================================================================
# SARSA
# =============================================================================

def sarsa(env, n_episodes=1000, max_steps=100,
          alpha=0.25, gamma=0.9, epsilon=0.15):
    """
    Tabular SARSA (on-policy TD) for the gene network (no terminal state).

    Returns
    -------
    Q               : np.ndarray (N, N_ACTS)
    policy          : np.ndarray (N,)
    episode_rewards : np.ndarray (n_episodes,)
    """
    N               = env['N']
    N_ACTS          = env['N_ACTS']
    Q               = np.zeros((N, N_ACTS))
    episode_rewards = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state        = np.random.randint(N)
        action       = _epsilon_greedy(Q, state, N_ACTS, epsilon)
        total_reward = 0.0

        for _ in range(max_steps):
            next_state, reward = _sample_step(env, state, action)
            next_action        = _epsilon_greedy(Q, next_state, N_ACTS, epsilon)

            # SARSA update
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            total_reward += reward
            state  = next_state
            action = next_action

        episode_rewards[ep] = total_reward

    policy = np.argmax(Q, axis=1)
    return Q, policy, episode_rewards
