import numpy as np


def _sample_step(env, state_idx, action):
    """Sample (next_state_idx, reward) for the gene network."""
    N           = env['N']
    next_idx    = np.random.choice(N, p=env['M'][action][state_idx])
    reward      = env['gene_reward'][next_idx] + env['action_cost'][action]
    return next_idx, reward



# Q-Learning
def q_learning(env, n_episodes=1000, max_steps=100,
               alpha=0.25, gamma=0.9, epsilon=0.15):

    N               = env['N']
    N_ACTS          = env['N_ACTS']
    Q               = np.zeros((N, N_ACTS))
    episode_rewards = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state        = np.random.randint(N)   # random start
        total_reward = 0.0

        for _ in range(max_steps):
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(N_ACTS)
            else:
                action = np.argmax(Q[state])

            next_state, reward = _sample_step(env, state, action)

            # Q-Learning update
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            total_reward += reward
            state         = next_state

        episode_rewards[ep] = total_reward

    policy = np.argmax(Q, axis=1)
    return Q, policy, episode_rewards
