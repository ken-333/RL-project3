# EECE 5614 — Project 3: Temporal Difference Learning

## Overview

This project implements temporal difference (TD) learning algorithms on two environments from Project 2:

- **Problem 1**: Stochastic maze navigation (Q-Learning, SARSA, Tabular Actor-Critic)
- **Problem 2**: p53-MDM2 gene network control (Q-Learning, SARSA, SARSA(λ), Tabular Actor-Critic)

---

## Directory Structure

```
project3/
├── problem1/
│   ├── maze_env.py          # Maze environment (reused from Project 2)
│   ├── visualize.py         # Plotting utilities
│   ├── q_learning.py        # Q-Learning algorithm
│   ├── sarsa.py             # SARSA algorithm
│   ├── actor_critic.py      # Tabular Actor-Critic algorithm
│   ├── run_experiments.py   # Main entry point for Problem 1
│   └── result/              # Output figures
└── problem2/
    ├── gene_env.py          # Gene network environment (modified from Project 2)
    ├── q_learning.py        # Q-Learning algorithm
    ├── sarsa.py             # SARSA algorithm
    ├── sarsa_lambda.py      # SARSA(λ) algorithm
    ├── actor_critic.py      # Tabular Actor-Critic algorithm
    ├── run_experiments.py   # Main entry point for Problem 2
    └── result/              # Output figures
```

---

## Parameters

### Problem 1 (Maze)

| Parameter | Value |
|-----------|-------|
| p (stochasticity) | 0.025 |
| γ (discount) | 0.96 |
| α (learning rate) | 0.25 |
| ε (exploration) | 0.1 |
| Episodes | 1,000 |
| Max steps per episode | 1,000 |
| β (actor-critic) | 0.05 |
| Independent runs | 10 |

### Problem 2 (Gene Network)

| Parameter | Value |
|-----------|-------|
| p (noise) | 0.1 |
| γ (discount) | 0.9 |
| α (learning rate) | 0.25 |
| ε (exploration) | 0.15 |
| Episodes | 1,000 |
| Max steps per episode | 100 |
| β (actor-critic) | 0.05 |
| λ (SARSA-λ) | 0.95 |
| Independent runs | 10 |

---

## Algorithms

### Q-Learning (off-policy TD)
```
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

### SARSA (on-policy TD)
```
Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
```

### SARSA(λ) (eligibility traces, Problem 2 only)
```
δ  ← r + γ Q(s',a') − Q(s,a)
e(s,a) ← e(s,a) + 1
Q(s,a) ← Q(s,a) + α δ e(s,a)   for all s, a
e(s,a) ← γ λ e(s,a)             for all s, a
```

### Tabular Actor-Critic
```
δt = R_{t+1} + γ V(s_{t+1}) − V(s_t)
V(s_t)    ← V(s_t)    + α δt
H(s_t,a_t) ← H(s_t,a_t) + β δt (1 − π(a_t|s_t))
π(a|s) = exp(H(s,a)) / Σ_{a'} exp(H(s,a'))
```

---

## How to Run

```bash
# Problem 1
cd problem1
python run_experiments.py

# Problem 2
cd problem2
python run_experiments.py
```

Figures are saved to the `result/` folder in each problem directory.

---

## Notes

- `maze_env.py` is copied directly from Project 2 with no modifications.
- `gene_env.py` is modified from Project 2: the action space is reduced to 4 actions (a1–a4), and the action cost is c(a1)=c(a4)=0, c(a2)=c(a3)=1.
- All algorithms use ε-greedy exploration during training and greedy policy for evaluation.
