import os
import numpy as np
import matplotlib.pyplot as plt

from maze_env import build_maze_env
from q_learning  import q_learning
from sarsa       import sarsa
from actor_critic import actor_critic
from visualize   import (plot_policy, plot_path,
                          plot_avg_cumulative_rewards,
                          plot_algorithm_comparison)

# Experiment parameters
P          = 0.025
GAMMA      = 0.96
ALPHA      = 0.25
EPSILON    = 0.1
BETA       = 0.05        # actor-critic only (default; tuned version uses 0.2)
N_EPISODES = 1000
MAX_STEPS  = 1000
N_RUNS     = 10  # independent runs

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(RESULT_DIR, exist_ok=True) # 确保结果目录存在，如果不存在则创建它


#  simulate one greedy trajectory (for policy / path plots)
def simulate_greedy(env, policy, max_steps=MAX_STEPS): # simulate_greedy 函数的作用是根据给定的环境和策略，模拟一个贪婪的轨迹（从起始状态出发，按照策略选择动作，直到达到目标状态或达到最大步数）。它返回一个包含访问过的状态列表。
    s2i   = env['state_to_idx']
    i2s   = env['idx_to_state']
    start = s2i[env['start_state']]
    goal  = s2i[env['goal_state']]
    N     = env['N']

    curr   = start
    states = [i2s[curr]]
    for _ in range(max_steps): # 模拟一个轨迹，最多执行 max_steps 步
        if curr == goal:  # 如果当前状态是目标状态，停止模拟
            break
        a        = policy[curr]  # 根据当前状态 curr，从策略 policy 中选择动作 a
        next_idx = np.random.choice(N, p=env['M'][a][curr]) # 根据环境的转移概率矩阵 M，随机选择下一个状态索引 next_idx
        states.append(i2s[next_idx])
        curr = next_idx
    return states



# run one algorithm for N_RUNS independent trials
def run_algorithm(algo_name, algo_fn, algo_kwargs, env): # run_algorithm 函数的作用是运行指定的强化学习算法（如 Q-Learning、SARSA 或 Actor-Critic）进行 N_RUNS 次独立试验，并收集每次试验的结果，包括策略、每集奖励、收敛集数和是否找到路径。它返回一个包含所有试验结果的列表和平均奖励曲线。
    """
    Run one algorithm for N_RUNS independent trials.

    Returns
    -------
    results : list of dicts, one per run, each with keys:
        'policy', 'episode_rewards', 'convergence_ep', 'path_found'
    avg_rewards : np.ndarray (N_EPISODES,)  mean episode reward over runs
    """
    s2i      = env['state_to_idx']
    goal_idx = s2i[env['goal_state']]

    n_eps        = algo_kwargs.get('n_episodes', N_EPISODES) # 获取算法参数中的 n_episodes，如果没有则使用默认值 N_EPISODES
    results      = []
    all_rewards  = np.zeros((N_RUNS, n_eps))

    print(f"\n  Running {algo_name} ({N_RUNS} independent runs)...")
    for run in range(N_RUNS):  # 进行 N_RUNS 次独立试验
        out = algo_fn(env, **algo_kwargs) # 根据传入的算法函数 algo_fn 和参数 algo_kwargs，运行强化学习算法，并获取输出结果 out。对于 Q-Learning 和 SARSA，out 包含 (Q, policy, rewards, conv_ep)；对于 Actor-Critic，out 包含 (V, H, policy, rewards, conv_ep)。

        # Unpack: actor_critic returns (V, H, policy, rewards, conv_ep)
        #         q_learning / sarsa return (Q, policy, rewards, conv_ep)
        if algo_name == 'Actor-Critic':  # 如果算法是 Actor-Critic，输出包含 V、H、policy、rewards 和 conv_ep
            _, _, policy, ep_rewards, conv_ep = out
        else:
            _, policy, ep_rewards, conv_ep = out

        # Check final path: simulate greedy from start
        traj       = simulate_greedy(env, policy)
        path_found = (s2i[traj[-1]] == goal_idx)

        results.append({   # 将每次试验的结果保存到 results 列表中，每个元素是一个字典，包含 'policy'、'episode_rewards'、'convergence_ep' 和 'path_found' 等键值对
            'policy':         policy,
            'episode_rewards': ep_rewards,
            'convergence_ep': conv_ep,
            'path_found':     path_found,
            'traj':           traj,
        })
        all_rewards[run] = ep_rewards  # 将每次试验的 episode_rewards 保存到 all_rewards 数组中，行索引对应 run，列索引对应 episode
        print(f"    run {run+1:2d}: conv_ep={conv_ep:5d}  "
              f"path_found={path_found}")

    avg_rewards = all_rewards.mean(axis=0) # 计算所有试验的平均 episode_rewards，得到 avg_rewards 数组，长度为 n_eps，每个元素是对应 episode 的平均奖励
    return results, avg_rewards


# Q1–Q5: three algorithms, fixed parameters
def run_q1_to_q5(env):
    print("\n" + "=" * 60)
    print("Problem 1 — Q1 to Q5")
    print("=" * 60)

    algo_configs = {  # 定义一个字典 algo_configs，包含三个算法的名称作为键，对应的值是一个元组，元组的第一个元素是算法函数（如 q_learning、sarsa 或 actor_critic），第二个元素是一个字典，包含该算法的参数（如 n_episodes、max_steps、alpha、gamma、epsilon 等）。这些参数将传递给 run_algorithm 函数，以运行每个算法并收集结果。
        'Q-Learning': (
            q_learning,
            dict(n_episodes=N_EPISODES, max_steps=MAX_STEPS,
                 alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
        ),
        'SARSA': (
            sarsa,
            dict(n_episodes=N_EPISODES, max_steps=MAX_STEPS,
                 alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
        ),
        'Actor-Critic': (
            actor_critic,
            dict(n_episodes=2000, max_steps=MAX_STEPS,
                 alpha=ALPHA, beta=0.2, gamma=GAMMA)
        ),
    }

    all_avg_rewards = {}   # for Q3 individual plots and Q4 comparison
    convergence_summary = {}

    for algo_name, (algo_fn, algo_kwargs) in algo_configs.items():
        results, avg_rewards = run_algorithm(
            algo_name, algo_fn, algo_kwargs, env)

        # Q1: success count
        n_success = sum(r['path_found'] for r in results)
        print(f"\n  [{algo_name}] Paths found: {n_success} / {N_RUNS}")

        #Q2: policy + path for one successful run
        successful = [r for r in results if r['path_found']]
        show_run   = successful[0] if successful else results[0]

        plot_policy(show_run['policy'], env,
                    title=f'Optimal Policy — {algo_name}',
                    save_path=os.path.join(
                        RESULT_DIR, f'policy_{algo_name.replace("-","_").lower()}.png'))
        plot_path(show_run['traj'], env,
                  title=f'Optimal Path — {algo_name}',
                  save_path=os.path.join(
                      RESULT_DIR, f'path_{algo_name.replace("-","_").lower()}.png'))

        # Q3: individual avg reward curve 
        plot_avg_cumulative_rewards(
            {algo_name: avg_rewards},
            title=f'Average Accumulated Reward — {algo_name}',
            save_path=os.path.join(
                RESULT_DIR, f'reward_{algo_name.replace("-","_").lower()}.png'))

        all_avg_rewards[algo_name] = avg_rewards

        #Q5: convergence episodes
        conv_eps = [r['convergence_ep'] for r in results]
        convergence_summary[algo_name] = conv_eps

    #Q4: all algorithms on one plot (truncate to N_EPISODES for fair comparison)
    comparison_rewards = {k: v[:N_EPISODES] for k, v in all_avg_rewards.items()}
    plot_algorithm_comparison(
        comparison_rewards,
        title='Average Accumulated Reward — All Algorithms',
        save_path=os.path.join(RESULT_DIR, 'reward_comparison.png'))

    # Q5: print convergence table
    print("\n" + "=" * 60)
    print("Q5 — Convergence Episode (greedy policy first finds path)")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Run 1-10 convergence episodes'}")
    print("-" * 60)
    for algo_name, conv_eps in convergence_summary.items():
        ep_str = "  ".join(f"{e:4d}" for e in conv_eps)
        mean   = np.mean(conv_eps)
        print(f"  {algo_name:<13} [{ep_str}]  mean={mean:.1f}")

    return all_avg_rewards


# Q6: alpha sensitivity analysis
def run_q6_alpha_sensitivity(env):
    print("\n" + "=" * 60)
    print("Q6 — Alpha Sensitivity (Q-Learning)")
    print("=" * 60)

    alpha_values = [0.05, 0.1, 0.25, 0.5]
    curves       = {}

    for alpha_val in alpha_values:
        all_rewards = np.zeros((N_RUNS, N_EPISODES))
        for run in range(N_RUNS):
            _, _, ep_rewards, _ = q_learning(
                env,
                n_episodes=N_EPISODES,
                max_steps=MAX_STEPS,
                alpha=alpha_val,
                gamma=GAMMA,
                epsilon=EPSILON,
            )
            all_rewards[run] = ep_rewards
        curves[f'α={alpha_val}'] = all_rewards.mean(axis=0)
        print(f"  α={alpha_val} done")

    plot_avg_cumulative_rewards(
        curves,
        title='Q-Learning — Effect of Learning Rate α',
        save_path=os.path.join(RESULT_DIR, 'alpha_sensitivity.png'))


# Main
if __name__ == '__main__':
    print("Building maze environment  (p=0.025)...")
    env = build_maze_env(p_stochastic=P)
    print(f"  States: {env['N']}  |  Start: {env['start_state']}  "
          f"|  Goal: {env['goal_state']}")

    run_q1_to_q5(env)
    run_q6_alpha_sensitivity(env)

    print("\nAll figures saved to:", RESULT_DIR)
