
import numpy as np
from matplotlib.patches import Rectangle

W = np.nan  # shorthand for wall (内部墙壁)

STATE_MATRIX = np.array([
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # row 0  外圈
    [0, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,   0],  # row 1
    [0, 214, 215, 216, 217,   W, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,   0],  # row 2
    [0, 197, 198, 199, 200,   W, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,   0],  # row 3
    [0, 193, 194,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W, 195, 196,   0],  # row 4
    [0, 176, 177,   W, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,   0],  # row 5
    [0, 162, 163,   W, 164, 165,   W, 166, 167,   W, 168, 169, 170, 171, 172,   W, 173, 174, 175,   0],  # row 6
    [0, 151, 152,   W, 153, 154,   W, 155, 156,   W, 157, 158,   W,   W,   W,   W, 159, 160, 161,   0],  # row 7
    [0, 136, 137, 138, 139, 140,   W, 141, 142,   W, 143, 144, 145, 146, 147,   W, 148, 149, 150,   0],  # row 8
    [0, 121, 122, 123, 124, 125,   W, 126, 127,   W, 128, 129, 130, 131, 132,   W, 133, 134, 135,   0],  # row 9
    [0,   W,   W,   W,   W, 111,   W, 112, 113,   W,   W, 114, 115, 116, 117,   W, 118, 119, 120,   0],  # row 10
    [0,  99, 100, 101, 102, 103,   W, 104, 105, 106,   W, 107, 108,   W, 109,   W,   W,   W, 110,   0],  # row 11
    [0,  89,  90,   W,   W,   W,   W,   W,  91,  92,   W,  93,  94,   W,  95,  96,  97,   W,  98,   0],  # row 12
    [0,  75,  76,  77,  78,  79,  80,   W,  81,  82,   W,  83,  84,   W,  85,  86,  87,   W,  88,   0],  # row 13
    [0,  60,  61,  62,  63,  64,  65,   W,  66,  67,   W,  68,  69,   W,  70,  71,  72,  73,  74,   0],  # row 14
    [0,  47,  48,  49,  50,  51,  52,   W,  53,  54,  55,  56,  57,   W,   W,   W,   W,  58,  59,   0],  # row 15
    [0,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,   0],  # row 16
    [0,   W,   W,  19,  20,  21,  22,   W,   W,   W,   W,   W,   W,  23,  24,  25,  26,  27,  28,   0],  # row 17
    [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,   0],  # row 18
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # row 19 外圈
], dtype=float)

# Special cells 
BUMP_CELLS = [        # 进入时额外扣 −10（碰撞惩罚）
    (1,11),(1,12),
    (2,1),(2,2),(2,3),
    (5,1),(5,9),(5,17),
    (6,17),
    (7,2),(7,10),(7,11),(7,17),
    (8,17),
    (12,11),(12,12),
    (14,1),(14,2),
    (15,17),(15,18),
    (16,7),
]

OIL_CELLS = [             # 进入时额外扣 −5（油滑惩罚）
    (2,8),(2,16),
    (4,2),
    (5,6),
    (10,18),
    (15,10),
    (16,10),
    (17,14),(17,17),
    (18,7),
]

START_CELL = (15, 4)   # state id = 50
GOAL_CELL  = (3, 13)   # state id = 208


# Actions:  0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  #action=0（Up）→ dr=-1, dc=0
ACTION_NAMES = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}

# Perpendicular action pairs  (for stochastic drift)
PERPENDICULAR = {
    0: [2, 3],   # Up    ↔ Left, Right
    1: [2, 3],   # Down  ↔ Left, Right
    2: [0, 1],   # Left  ↔ Up,   Down
    3: [0, 1],   # Right ↔ Up,   Down
}

def rc_to_state(row, col):  # 坐标 → state id（墙返回 None）
    """(row, col) → state id, or None if wall/border."""
    val = STATE_MATRIX[row, col]
    if np.isnan(val) or val == 0:  #如果是 nan（内部墙壁）或 0（边界）→ 返回 None，表示"这不是合法状态"
        return None  
    return int(val)   #否则返回那个正整数，就是 state id（1–248）

def state_to_rc(state_id):   # 在整个 20×20 矩阵里搜索哪个位置的值等于 state_id，返回所有匹配位置的坐标数组
    """state id → (row, col)."""
    pos = np.argwhere(STATE_MATRIX == state_id)
    if len(pos) == 0:
        return None
    return tuple(pos[0])

def next_cell(row, col, action): 
    """
    Apply *action* from (row, col).
    If the resulting cell is a wall or border the agent stays put.
    Returns (new_row, new_col).
    """
    dr, dc = ACTIONS[action]  
    nr, nc = row + dr, col + dc  
    if nr < 1 or nr > 18 or nc < 1 or nc > 18:  
        return row, col         #检查目标坐标是否越出了内部网格范围（合法范围是 1–18）。如果越界了，说明撞到了外墙，agent 原地不动，返回原坐标。                   
    
    val = STATE_MATRIX[nr, nc]       
    if np.isnan(val) or val == 0:              
        return row, col                          
    return nr, nc                         


# Build state sets (called once and cached inside build_maze_env)
def build_state_sets():
    """
    Returns
    -------
    all_states  : sorted list of valid state ids  [1 … 248]
    bump_states : set of bump state ids
    oil_states  : set of oil  state ids
    start_state : state id of start cell
    goal_state  : state id of goal  cell
    """
    all_states = sorted(     #遍历内部 18×18 网格（row 1–18, col 1–18），把所有不是 nan 也不是 0 的值取出来转成整数，排序后得到一个列表 [1, 2, 3, ..., 248]。这就是所有合法状态的 id。
        int(STATE_MATRIX[r, c])
        for r in range(1, 19) for c in range(1, 19)
        if not np.isnan(STATE_MATRIX[r, c]) and STATE_MATRIX[r, c] != 0
    )
    bump_states = {rc_to_state(r, c) for (r, c) in BUMP_CELLS   #遍历之前定义的 BUMP_CELLS 列表里的每个坐标，用 rc_to_state 转成 state id，收集成一个集合（set）。
                   if rc_to_state(r, c) is not None} 
    oil_states  = {rc_to_state(r, c) for (r, c) in OIL_CELLS 
                   if rc_to_state(r, c) is not None}
    start_state = rc_to_state(*START_CELL) 
    goal_state  = rc_to_state(*GOAL_CELL) 
    return all_states, bump_states, oil_states, start_state, goal_state


# Core builder — produces M(a) and R_s^a
def build_maze_env(p_stochastic, bump_penalty=-10):
    
    all_states, bump_states, oil_states, start_state, goal_state = build_state_sets()  

    N      = len(all_states)                                  # 248
    s2i    = {s: i for i, s in enumerate(all_states)}        # state id → idx  
    i2s    = {i: s for i, s in enumerate(all_states)}        # idx → state id


    # 1.  Build M(a)  —  four N×N transition matrices
    M = [np.zeros((N, N)) for _ in range(4)]     
    for s in all_states:
        si       = s2i[s] 
        row, col = state_to_rc(s) 

        # Goal is absorbing: stays there with probability 1 
        if s == goal_state:
            for a in range(4):
                M[a][si, si] = 1.0
            continue 

        for a in range(4):
            # main direction: prob (1 − p)
            mr, mc = next_cell(row, col, a)
            mi     = s2i[rc_to_state(mr, mc)] 

            #  two perpendicular directions: prob p/2 each
            pa1, pa2 = PERPENDICULAR[a]
            pr1, pc1 = next_cell(row, col, pa1)
            pr2, pc2 = next_cell(row, col, pa2)
            pi1 = s2i[rc_to_state(pr1, pc1)] 
            pi2 = s2i[rc_to_state(pr2, pc2)]


            M[a][si, mi]  += (1.0 - p_stochastic) 
            M[a][si, pi1] += p_stochastic / 2.0  
            M[a][si, pi2] += p_stochastic / 2.0  

    # 2.  Build R(s, a, s')  — immediate reward for each (s, a, s') triple
    R_full = np.zeros((N, 4, N))    # R_full[si, a, sj]

    for s in all_states:
        si       = s2i[s] #状态编号 → 矩阵下标
        row, col = state_to_rc(s)

        if s == goal_state:
            # Absorbing state: no further reward
            continue

        for a in range(4):
            # Determine whether the *main* direction hits a wall
            mr, mc        = next_cell(row, col, a)
            main_hit_wall = (mr == row and mc == col) #如果主方向的落点坐标和原坐标相同，说明主方向撞墙了，main_hit_wall 就是 True；否则就是 False。

            # Determine whether each *perpendicular* direction hits a wall
            pa1, pa2      = PERPENDICULAR[a]  # 取出两个垂直动作
            pr1, pc1      = next_cell(row, col, pa1)
            pr2, pc2      = next_cell(row, col, pa2)
            perp1_hit     = (pr1 == row and pc1 == col)  # 垂直方向1是否撞墙 Ture/False
            perp2_hit     = (pr2 == row and pc2 == col)  # 垂直方向2是否撞墙

            # Map destination cells to state indices
            mi  = s2i[rc_to_state(mr,  mc)]
            pi1 = s2i[rc_to_state(pr1, pc1)]
            pi2 = s2i[rc_to_state(pr2, pc2)] #我们已经计算了主方向和两个垂直方向的落点坐标，现在将它们转换成状态编号，再转换成矩阵下标 mi, pi1, pi2。

            # Assign R(s, a, s') for each reachable s' 
            destinations = [
                (mi,  main_hit_wall), # 主方向落点，是否撞墙
                (pi1, perp1_hit),   # 垂直方向1落点，是否撞墙
                (pi2, perp2_hit),   # 垂直方向2落点，是否撞墙
            ]
            #计算每个落点的奖励
            for (sj, hit_wall) in destinations:
                sp   = i2s[sj]     # 矩阵下标 → 状态编号
                base = -1.0                        # action cost (always)

                if hit_wall:
                    base += -0.8                   # wall penalty

                if sp in oil_states:
                    base += -5.0

                if sp in bump_states:
                    base += bump_penalty           # default −10

                if sp == goal_state:
                    base += 200.0


                R_full[si, a, sj] = base

    # 3.  Build R_s^a
    R_sa = np.zeros((N, 4)) 
    for a in range(4):
        # element-wise product M[a] ⊙ R_full[:,a,:]  then sum over j
        R_sa[:, a] = np.sum(M[a] * R_full[:, a, :], axis=1) #R_sa[:, a]取出第a列的所有元素        右边算出一个 (248,) 的向量，直接整列写入 R_sa 的第 a 列，比逐行赋值高效得多。
        #M[a] 形状 (248, 248)，概率矩阵   R_full[:, a, :] 形状 (248, 248)，奖励矩阵 是从三维矩阵里**切出固定动作a的一层**

    return {
        'M'            : M,            # list of 4  N×N matrices
        'R_sa'         : R_sa,         # N×4  expected reward      
        'R_full'       : R_full,       # N×4×N raw reward (for simulation)
        'N'            : N,
        'all_states'   : all_states,
        'state_to_idx' : s2i,
        'idx_to_state' : i2s,
        'start_state'  : start_state,
        'goal_state'   : goal_state,
        'bump_states'  : bump_states,
        'oil_states'   : oil_states,
    }


# Quick sanity-check
if __name__ == '__main__':
    all_states, bump_states, oil_states, start_state, goal_state = \
        build_state_sets()

    print(f"Total states : {len(all_states)}")    # expected: 248
    print(f"Bump states  : {len(bump_states)}")   # expected: 21
    print(f"Oil  states  : {len(oil_states)}")    # expected: 10
    print(f"Start state  : {start_state}")        # expected: 50
    print(f"Goal  state  : {goal_state}")         # expected: 208

    env = build_maze_env(p_stochastic=0.02)

    M    = env['M'] 
    R_sa = env['R_sa']

    print(f"\nM[0] shape   : {M[0].shape}")       # (248, 248) 0=Up, 1=Down, 2=Left, 3=Right
    print(f"R_sa shape   : {R_sa.shape}")          # (248, 4)

    # Each row of every M[a] must sum to 1
    for a in range(4):
        row_sums = M[a].sum(axis=1)
        print(f"M[{a}] row-sum range: "
              f"[{row_sums.min():.6f}, {row_sums.max():.6f}]")  # both ≈ 1

    # Verify start / goal positions
    print(f"\nStart ({start_state}) @ {state_to_rc(start_state)}")
    print(f"Goal  ({goal_state})  @ {state_to_rc(goal_state)}")

    # Example: reproduce TA reward examples from the handout (small grid)
    # R(4, L, 3) = -6  →  check conceptually with the actual maze
    si = env['state_to_idx'][start_state]
    print(f"\nR_sa[start, Left] = {R_sa[si, 2]:.4f}")


