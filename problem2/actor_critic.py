import numpy as np

def _sample_step(env, state_idx, action):
    """Sample (next_state_idx, reward) for the gene network."""
    N        = env['N']
    next_idx = np.random.choice(N, p=env['M'][action][state_idx])
    reward   = env['gene_reward'][next_idx] + env['action_cost'][action]
    return next_idx, reward


def _softmax_policy(H, state_idx):
    """π(·|s) = softmax(H[s,:]). Returns probability vector (N_ACTS,)."""
    h     = H[state_idx] - np.max(H[state_idx])   # numerical stability
    exp_h = np.exp(h)
    return exp_h / exp_h.sum()


# Tabular Actor-Critic
def actor_critic(env, n_episodes=1000, max_steps=100,
                 alpha=0.25, beta=0.05, gamma=0.9):
    """
    Tabular Actor-Critic for the gene network (no terminal state).

    Critic : V(s)    updated with TD error δ
    Actor  : H(s,a)  preference table, π(a|s) = softmax(H(s,:))

    Returns
    -------
    V               : np.ndarray (N,)
    H               : np.ndarray (N, N_ACTS)
    policy          : np.ndarray (N,)   argmax of H
    episode_rewards : np.ndarray (n_episodes,)
    """
    N               = env['N']
    N_ACTS          = env['N_ACTS']
    V               = np.zeros(N)
    H               = np.zeros((N, N_ACTS))
    episode_rewards = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state        = np.random.randint(N)
        total_reward = 0.0

        for _ in range(max_steps):
            pi     = _softmax_policy(H, state)
            action = np.random.choice(N_ACTS, p=pi)

            next_state, reward = _sample_step(env, state, action)

            # TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Critic update
            V[state] += alpha * delta

            # Actor update
            H[state, action] += beta * delta * (1.0 - pi[action])

            total_reward += reward
            state = next_state

        episode_rewards[ep] = total_reward

    policy = np.argmax(H, axis=1)
    return V, H, policy, episode_rewards
