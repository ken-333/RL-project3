import numpy as np


def _sample_step(env, state_idx, action):  #输入三个参数：环境字典 env，当前状态的矩阵下标 state_idx，要执行的动作 action（0/1/2/3）。
    """Sample (next_state_idx, reward) from the stochastic environment."""
    N      = env['N']     # 取出状态总数，248。
    next_idx = np.random.choice(N, p=env['M'][action][state_idx])
    reward   = env['R_full'][state_idx, action, next_idx]
    return next_idx, reward


def _greedy_path_found(Q, env, max_steps):
    """
    Check whether the greedy policy derived from Q finds a path
    from start to goal within max_steps (stochastic simulation).
    """
    s2i      = env['state_to_idx']
    start    = s2i[env['start_state']]
    goal     = s2i[env['goal_state']]
    curr     = start
    for _ in range(max_steps):
        if curr == goal:
            return True
        action = np.argmax(Q[curr])
        curr, _ = _sample_step(env, curr, action)
    return curr == goal



# Q-Learning
def q_learning(env, n_episodes=1000, max_steps=1000,
               alpha=0.25, gamma=0.96, epsilon=0.1):
    
    N         = env['N']
    s2i       = env['state_to_idx']
    start_idx = s2i[env['start_state']]
    goal_idx  = s2i[env['goal_state']]

    Q               = np.zeros((N, 4)) # 初始化Q表，行数为状态数N，列数为动作数4（上、下、左、右）。
    episode_rewards = np.zeros(n_episodes)
    convergence_ep  = n_episodes  # 先设成 1000，表示"还没收敛"。后面一旦找到路径就会更新。

    for ep in range(n_episodes):  #外层循环（遍历每个 episode）
        state        = start_idx
        total_reward = 0.0

        for _ in range(max_steps):
            if state == goal_idx:
                break   # 如果当前状态已经是目标状态，就结束这个 episode 的循环。

            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state])

            next_state, reward = _sample_step(env, state, action)

            # Q-Learning update: off-policy (uses max over next actions)
            Q[state, action] += alpha * (  # Q[state, action]：Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            total_reward += reward
            state         = next_state

        episode_rewards[ep] = total_reward

        # Track first episode where greedy policy reaches goal
        if convergence_ep == n_episodes:
            if _greedy_path_found(Q, env, max_steps):
                convergence_ep = ep + 1  # 1-indexed

    policy = np.argmax(Q, axis=1)  # 从 Q 表中提取贪婪策略：对于每个状态，选择具有最高 Q 值的动作。
    return Q, policy, episode_rewards, convergence_ep
