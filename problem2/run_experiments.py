import os
import numpy as np
import matplotlib.pyplot as plt

from gene_env     import build_gene_env, ALL_STATES
from q_learning   import q_learning
from sarsa        import sarsa
from sarsa_lambda import sarsa_lambda
from actor_critic import actor_critic


# Experiment parameters
P          = 0.1
GAMMA      = 0.9
ALPHA      = 0.25
EPSILON    = 0.15
BETA       = 0.05     # actor-critic only
LAM        = 0.95     # SARSA(λ) default
N_EPISODES = 1000
MAX_STEPS  = 100
N_RUNS     = 10

ACTION_NAMES  = ['a1', 'a2', 'a3', 'a4']
STATE_LABELS  = [''.join(str(int(b)) for b in ALL_STATES[i]) for i in range(16)]

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)


# evaluate state visitation under greedy policy
def evaluate_state_visits(env, policy, n_eval_episodes=100, episode_length=100):
    """
    Run greedy policy for n_eval_episodes × episode_length steps.
    Returns visits: np.ndarray (N,) — count of how many times each state visited.
    """
    N      = env['N']
    visits = np.zeros(N, dtype=int)
    for _ in range(n_eval_episodes):
        state = np.random.randint(N)
        for _ in range(episode_length):
            visits[state] += 1
            action = policy[state]
            state  = np.random.choice(N, p=env['M'][action][state])
    return visits



#run one algorithm for N_RUNS independent trials
def run_algorithm(algo_name, algo_fn, algo_kwargs, env):
    n_eps       = algo_kwargs.get('n_episodes', N_EPISODES)
    results     = []
    all_rewards = np.zeros((N_RUNS, n_eps))

    print(f"\n  Running {algo_name} ({N_RUNS} independent runs)...")
    for run in range(N_RUNS):
        out = algo_fn(env, **algo_kwargs)

        # Unpack based on algorithm type
        if algo_name == 'Actor-Critic':
            _, _, policy, ep_rewards = out
        else:
            _, policy, ep_rewards = out

        visits = evaluate_state_visits(env, policy)

        results.append({
            'policy':          policy,
            'episode_rewards': ep_rewards,
            'visits':          visits,
        })
        all_rewards[run] = ep_rewards
        print(f"    run {run+1:2d}: mean_ep_reward={ep_rewards.mean():.2f}")

    avg_rewards = all_rewards.mean(axis=0)
    return results, avg_rewards


def _smooth(curve, window=50):
    """Rolling mean over `window` episodes for cleaner visualization."""
    if window <= 1:
        return curve
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode='same')


def plot_reward_curve(curves_dict, title, save_path, smooth_window=50):
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, curve), color in zip(curves_dict.items(), colors):
        ax.plot(_smooth(curve, smooth_window), label=label, color=color)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Accumulated Reward')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_state_visits(visits_dict, title, save_path):
    """Bar chart of state visitation counts for multiple algorithms."""
    n_algos = len(visits_dict)
    fig, axes = plt.subplots(1, n_algos, figsize=(4 * n_algos, 5), sharey=True)
    if n_algos == 1:
        axes = [axes]
    for ax, (algo_name, visits) in zip(axes, visits_dict.items()):
        ax.bar(STATE_LABELS, visits, color='steelblue', edgecolor='black', lw=0.5)
        ax.set_title(algo_name)
        ax.set_xlabel('State')
        ax.tick_params(axis='x', rotation=90)
    axes[0].set_ylabel('Visit Count')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Q1–Q4: all four algorithms
def run_q1_to_q4(env):
    print("\n" + "=" * 60)
    print("Problem 2 — Q1 to Q4")
    print("=" * 60)

    algo_configs = {
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
        'SARSA-lambda': (
            sarsa_lambda,
            dict(n_episodes=N_EPISODES, max_steps=MAX_STEPS,
                 alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, lam=LAM)
        ),
        'Actor-Critic': (
            actor_critic,
            dict(n_episodes=N_EPISODES, max_steps=MAX_STEPS,
                 alpha=ALPHA, beta=BETA, gamma=GAMMA)
        ),
    }

    all_avg_rewards = {}
    all_visits      = {}

    for algo_name, (algo_fn, algo_kwargs) in algo_configs.items():
        results, avg_rewards = run_algorithm(algo_name, algo_fn, algo_kwargs, env)

        # Q1: optimal policy for all 10 runs
        print(f"\n  [{algo_name}] Optimal policies (10 runs):")
        print(f"  {'State':<8}", end='')
        for r in range(N_RUNS):
            print(f"  run{r+1:<2}", end='')
        print()
        print(f"  {'-'*70}")
        for i in range(16):
            print(f"  {STATE_LABELS[i]:<8}", end='')
            for r in range(N_RUNS):
                print(f"  {ACTION_NAMES[results[r]['policy'][i]]:<5}", end='')
            print()

        #Q2: individual avg reward curve
        plot_reward_curve(
            {algo_name: avg_rewards},
            title=f'Average Accumulated Reward — {algo_name}',
            save_path=os.path.join(
                RESULT_DIR,
                f'reward_{algo_name.lower().replace("-","_").replace(" ","_")}.png'
            )
        )

        all_avg_rewards[algo_name] = avg_rewards

        # Aggregate visits (average over 10 runs)
        avg_visits = np.mean([r['visits'] for r in results], axis=0).astype(int)
        all_visits[algo_name] = avg_visits

    # Q3: all algorithms comparison plot
    plot_reward_curve(
        all_avg_rewards,
        title='Average Accumulated Reward — All Algorithms',
        save_path=os.path.join(RESULT_DIR, 'reward_comparison.png')
    )
    print("\n  Comparison plot saved.")

    #Q4: state visitation counts
    plot_state_visits(
        all_visits,
        title='State Visitation Counts (100 eval episodes per algorithm)',
        save_path=os.path.join(RESULT_DIR, 'state_visits.png')
    )
    print("  State visitation plot saved.")

    # Print visitation vectors
    print("\n  State Visitation Counts (averaged over 10 runs):")
    print(f"  {'State':<8}", end='')
    for name in all_visits:
        print(f"  {name:<14}", end='')
    print()
    for i in range(16):
        print(f"  {STATE_LABELS[i]:<8}", end='')
        for visits in all_visits.values():
            print(f"  {visits[i]:<14}", end='')
        print()

    return all_avg_rewards



# Q5: SARSA(λ) — effect of λ
def run_q5_lambda_sensitivity(env):
    print("\n" + "=" * 60)
    print("Q5 — SARSA(λ): Effect of λ")
    print("=" * 60)

    lambda_values = [0.0, 0.5, 0.95]
    curves        = {}

    for lam_val in lambda_values:
        all_rewards = np.zeros((N_RUNS, N_EPISODES))
        for run in range(N_RUNS):
            _, _, ep_rewards = sarsa_lambda(
                env,
                n_episodes=N_EPISODES,
                max_steps=MAX_STEPS,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
                lam=lam_val,
            )
            all_rewards[run] = ep_rewards
        curves[f'λ={lam_val}'] = all_rewards.mean(axis=0)
        print(f"  λ={lam_val} done")

    plot_reward_curve(
        curves,
        title='SARSA(λ) — Effect of λ',
        save_path=os.path.join(RESULT_DIR, 'lambda_sensitivity.png')
    )



# Main
if __name__ == '__main__':
    print("Building gene network environment  (p=0.1)...")
    env = build_gene_env(p_noise=P)
    print(f"  N={env['N']}, N_ACTS={env['N_ACTS']}")
    print(f"  action_cost={env['action_cost']}")

    run_q1_to_q4(env)
    run_q5_lambda_sensitivity(env)

    print("\nAll figures saved to:", RESULT_DIR)
