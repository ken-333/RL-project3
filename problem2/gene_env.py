import numpy as np

# Connectivity matrix C (4×4)
C = np.array([
    [ 0,  0, -1,  0],
    [ 1,  0, -1, -1],
    [ 0,  1,  0,  0],
    [-1,  1,  1,  0],
], dtype=float)

# All 16 states as binary vectors, shape (16, 4)
# s^1 = [0,0,0,0], s^2 = [0,0,0,1], ..., s^16 = [1,1,1,1]
# Index i corresponds to state s^{i+1} in the handout (0-indexed internally)
ALL_STATES = np.array([[int(b) for b in format(i, '04b')] for i in range(16)], dtype=float) #(16, 4) 的矩阵，每行是一个状态向量
#`format(i, '04b')` 把整数 i 转成4位二进制字符串，比如 `format(3, '04b') = '0011'`，对应状态 `[0,0,1,1]`。
# ALL_STATES[i] = binary vector for state index i
# ALL_STATES[0]  = [0,0,0,0]  ← s^1
# ALL_STATES[15] = [1,1,1,1]  ← s^16

# All 4 actions as binary vectors, shape (4, 4)
# Project 3 action space: a1=no control, a2=perturb p53, a3=perturb WIP1, a4=perturb MDM2
# Cost: c(a1)=c(a4)=0,  c(a2)=c(a3)=1
ALL_ACTIONS = np.array([
    [0, 0, 0, 0],  # a1: no control          cost=0
    [0, 1, 0, 0],  # a2: perturb p53         cost=1
    [0, 0, 1, 0],  # a3: perturb WIP1        cost=1
    [0, 0, 0, 1],  # a4: perturb MDM2        cost=0
], dtype=float)

# Action costs: c(a1)=0, c(a2)=1, c(a3)=1, c(a4)=0
ACTION_COST = np.array([0.0, -1.0, -1.0, 0.0])

N       = 16   # number of states
N_ACTS  = 4    # number of actions



#  threshold map  v̄
#   Maps each element of a vector to 1 if > 0, else 0
def threshold(v): # 这实现了题目里的 vˉ\bar{v} vˉ 算子：大于0的元素变成1，否则变成0。注意是 严格大于0，所以0本身映射到0，负数也映射到0。
    """v̄ operator: element-wise, 1 if v[i] > 0 else 0."""
    return (v > 0).astype(float) #在 NumPy 里，对一个数组做比较运算，会返回一个布尔数组.每个元素单独判断：大于0就是 True，否则是 False.  .astype(float) 把布尔值转成浮点数, True 变成 1.0，False 变成 0.0。最终返回一个和输入 v 形状相同的数组，每个元素是 1.0 或 0.0，表示 vˉ 的结果。


# Build gene network MDP
def build_gene_env(p_noise):

    # Index mappings
    s2i = {tuple(ALL_STATES[i].astype(int)): i for i in range(N)} #s2i：状态向量 → 索引，比如 s2i[(0,0,1,1)] 返回 3
    i2s = {i: ALL_STATES[i] for i in range(N)} #索引 → 状态向量，比如 i2s[3] 返回 array([0,0,1,1])

    # 1. Build M(a) — five N×N transition matrices
    M = [np.zeros((N, N)) for _ in range(N_ACTS)] # M[a] 是一个 N×N 的矩阵，表示在动作 a 下的状态转移概率。M[a][i,j] = P(s'=j | s=i, a)。

    for a_idx in range(N_ACTS): # 遍历5个动作
        a_vec = ALL_ACTIONS[a_idx]          # shape (4,)
        for i in range(N):       # 遍历16个当前状态
            s_i = ALL_STATES[i]             # shape (4,)

            # Step 1: C @ s_i  (continuous, may have negative values)
            Cs = C @ s_i                    # shape (4,)

            # Step 2: XOR with action  →  v̄(Cs) ⊕ a
            # Note: XOR is applied AFTER the threshold map
            expected = (threshold(Cs) + a_vec) % 2   # shape (4,)
            # expected[k] = (v̄(Cs)[k] + a[k]) mod 2

            # Step 3: fill row i of M[a] 对每个 j 计算转移概率
            for j in range(N):
                s_j  = ALL_STATES[j]        # shape (4,)
                diff = int(np.sum(np.abs(s_j - expected)))   # L1 distance diff 越大（需要翻转的位数越多），概率越小
                M[a_idx][i, j] = (p_noise ** diff) * ((1 - p_noise) ** (4 - diff))

    # 2. Build R(s, a, s') and R_sa
    # Precompute gene activation reward for each next state j: 5 * ||s^j||_1
    gene_reward = 5.0 * ALL_STATES.sum(axis=1)   # shape (N,)

    # Action costs: c(a1)=0, c(a2)=1, c(a3)=1, c(a4)=0  (stored as negative for addition)
    action_cost = ACTION_COST.copy()              # shape (N_ACTS,)

    R_sa = np.zeros((N, N_ACTS))
    for a_idx in range(N_ACTS):
        # R(s^i, a, s^j) = gene_reward[j] + action_cost[a_idx]
        R_sa[:, a_idx] = M[a_idx] @ gene_reward + action_cost[a_idx]

    return {
        'M'            : M,
        'R_sa'         : R_sa,
        'gene_reward'  : gene_reward,   # for TD step: reward = gene_reward[s'] + action_cost[a]
        'action_cost'  : action_cost,   # for TD step
        'N'            : N,
        'N_ACTS'       : N_ACTS,
        'all_states'   : ALL_STATES,
        'all_actions'  : ALL_ACTIONS,
        'state_to_idx' : s2i,
        'idx_to_state' : i2s,
    }


# Quick sanity-check
if __name__ == '__main__':
    env = build_gene_env(p_noise=0.1)
    M    = env['M']
    R_sa = env['R_sa']

    print(f"N = {env['N']},  N_ACTS = {env['N_ACTS']}")
    print(f"M[0] shape   : {M[0].shape}")     # (16, 16)
    print(f"R_sa shape   : {R_sa.shape}")     # (16, 4)
    print(f"gene_reward  : {env['gene_reward']}")
    print(f"action_cost  : {env['action_cost']}")

    for a in range(N_ACTS):
        row_sums = M[a].sum(axis=1)
        print(f"M[{a}] row-sum range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")