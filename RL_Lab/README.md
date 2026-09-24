# D.K.T.E. Society’s Textile and Engineering Institute, Ichalkaranji
### (An Autonomous Institute)
### Department of Computer Science & Engineering (Artificial Intelligence and Machine Learning)
**Class / Program:** Final Year B.Tech. (AI)  
**Semester:** VII  
**Course Code:** 01AMP403  
**Course Title:** Reinforcement Learning Lab  
**CIE Marks:** 50  

---

## Lab Curriculum Overview

This repository contains the complete laboratory practical implementations, mathematical formulations, experimental evaluations, visual analytical plots, and viva-voce resources for all 10 prescribed experiments of the Reinforcement Learning Lab curriculum.

Each experiment is housed in its dedicated modular directory (`exp1` through `exp10`) and is fully equipped with:
- Production-quality, thoroughly documented Python source code.
- Recorded console outputs and tabular performance statistics in `output.txt`.
- High-resolution comparative visualization charts (`.png`).
- An exhaustive academic `README.md` covering Aims, Theory, Algorithms, Execution Guides, Inferences, and Viva-Voce Questions with Answers.

---

## Index of Experiments

| Exp No. | Experiment Title | Core Focus / Tasks | Primary Environment | Directory Link |
| :---: | :--- | :--- | :--- | :---: |
| **01** | **Multi-Armed Bandit Problem** | $\varepsilon$-greedy action selection, Pure Greedy vs. $\varepsilon$-greedy vs. Decaying $\varepsilon$ vs. UCB1 exploration strategies. | 10-Armed Gaussian Testbed | [`exp1/`](./exp1/) |
| **02** | **Markov Decision Process (MDP) Modeling** | Engineering formulation (Textile AGV Material Transport), states, actions, transition matrices, rewards, analytical Bellman matrix inversion $V = (I - \gamma P)^{-1} R$. | Autonomous AGV Transport MDP | [`exp2/`](./exp2/) |
| **03** | **Dynamic Programming – Policy Evaluation** | Iterative policy evaluation for equiprobable random policy, Bellman expectation contraction, value matrix progression. | Sutton & Barto 4x4 Grid World | [`exp3/`](./exp3/) |
| **04** | **Policy Iteration** | Generalized Policy Iteration (GPI), alternating policy evaluation and greedy improvement, obstacle and trap avoidance. | 4x4 Grid World with Obstacles & Traps | [`exp4/`](./exp4/) |
| **05** | **Value Iteration** | Bellman optimality updates, contraction mapping, side-by-side benchmark with Policy Iteration (speed & sweeps). | 4x4 Grid World with Obstacles & Traps | [`exp5/`](./exp5/) |
| **06** | **Monte Carlo Prediction and Control** | First-Visit vs. Every-Visit MC value estimation, model-free On-Policy First-Visit MC Control with $\varepsilon$-greedy exploration. | Stochastic 4x4 Episodic Grid World | [`exp6/`](./exp6/) |
| **07** | **Temporal Difference (TD) Learning** | $TD(0)$ online bootstrapping prediction, empirical RMSE evaluation against analytical ground-truth values vs. Monte Carlo. | 5-State Random Walk MRP | [`exp7/`](./exp7/) |
| **08** | **SARSA Algorithm** | On-Policy TD Control ($S, A, R, S', A'$), $\varepsilon$-greedy decay, safety vs. risk aversion analysis. | Gymnasium CliffWalking-v1 | [`exp8/`](./exp8/) |
| **09** | **Q-Learning Algorithm & Comparison** | Off-Policy TD Control ($S, A, R, S'$), Bellman optimality, comparison with SARSA (Optimal cliff-hugging vs. Safe path; FrozenLake). | CliffWalking-v1 & FrozenLake-v1 | [`exp9/`](./exp9/) |
| **10** | **Deep Q-Network (DQN)** | Deep Q-Network in PyTorch, Experience Replay Buffer, Target Network, Continuous state control vs. Discretized Tabular baseline. | Gymnasium CartPole-v1 | [`exp10/`](./exp10/) |

---

## Directory Structure

```text
RL_Lab/
├── README.md                      <- Master Course Index & Laboratory Guide
├── run_all.py                     <- Master Test Runner Orchestrator
├── exp1/                          <- Multi-Armed Bandit
│   ├── bandit.py
│   ├── bandit_comparison.png
│   ├── output.txt
│   └── README.md
├── exp2/                          <- Markov Decision Process (MDP) Modeling
│   ├── mdp_model.py
│   ├── mdp_network_and_values.png
│   ├── output.txt
│   └── README.md
├── exp3/                          <- Dynamic Programming: Policy Evaluation
│   ├── policy_evaluation.py
│   ├── policy_evaluation_grid.png
│   ├── output.txt
│   └── README.md
├── exp4/                          <- Policy Iteration
│   ├── policy_iteration.py
│   ├── policy_iteration_result.png
│   ├── output.txt
│   └── README.md
├── exp5/                          <- Value Iteration
│   ├── value_iteration.py
│   ├── value_vs_policy_iteration.png
│   ├── output.txt
│   └── README.md
├── exp6/                          <- Monte Carlo Prediction & Control
│   ├── monte_carlo.py
│   ├── mc_prediction_control.png
│   ├── output.txt
│   └── README.md
├── exp7/                          <- Temporal Difference (TD) Learning
│   ├── td_learning.py
│   ├── td_vs_mc_comparison.png
│   ├── output.txt
│   └── README.md
├── exp8/                          <- SARSA Algorithm
│   ├── sarsa.py
│   ├── sarsa_cliffwalking.png
│   ├── output.txt
│   └── README.md
├── exp9/                          <- Q-Learning Algorithm
│   ├── q_learning.py
│   ├── q_learning_vs_sarsa.png
│   ├── output.txt
│   └── README.md
└── exp10/                         <- Deep Q-Network (DQN)
    ├── dqn_cartpole.py
    ├── dqn_cartpole_comparison.png
    ├── output.txt
    └── README.md
```

---

## Environment Setup & Prerequisites

All experiments are engineered in Python 3.10+ using standard scientific computing and reinforcement learning packages:

```bash
pip install numpy matplotlib torch gymnasium
```

---

## How to Run Experiments

### 1. Run Master Test Runner (All 10 Experiments)
Execute all experiments sequentially and display a consolidated status report:
```bash
python run_all.py
```

### 2. Run an Individual Experiment by Number
```bash
python run_all.py 1     # Runs Experiment 1
python run_all.py 8     # Runs Experiment 8
python run_all.py 10    # Runs Experiment 10
```

### 3. Run Directly From Experiment Folder
```bash
python exp1/bandit.py
python exp2/mdp_model.py
python exp3/policy_evaluation.py
python exp4/policy_iteration.py
python exp5/value_iteration.py
python exp6/monte_carlo.py
python exp7/td_learning.py
python exp8/sarsa.py
python exp9/q_learning.py
python exp10/dqn_cartpole.py
```

---

## Lab Viva-Voce Quick Revision Highlights

1. **Exploration vs Exploitation**: Balancing knowledge gathering (discovering new high-value actions) and knowledge exploiting (capitalizing on the best known action).
2. **Bellman Expectation vs. Optimality Equations**: Expectation averages over current policy probabilities; Optimality uses the $\max_a$ operator to represent the upper-bound performance achievable by an optimal policy.
3. **Model-Based vs. Model-Free**:
   - *Model-Based (Dynamic Programming)*: Requires explicit transition probabilities $\mathcal{P}(s'|s,a)$ and rewards $\mathcal{R}(s,a)$.
   - *Model-Free (Monte Carlo & TD)*: Learns directly through sampled experience trajectories.
4. **Monte Carlo vs. Temporal Difference**:
   - *Monte Carlo*: Zero bias, high variance, updates only at episode termination ($G_t$).
   - *Temporal Difference (TD)*: Low variance, small initial bias, updates online step-by-step by bootstrapping from successor state estimates ($R + \gamma V(S')$).
5. **On-Policy vs. Off-Policy Control**:
   - *On-Policy (SARSA)*: Target policy is the same as the behavior policy; learns the value of the exploratory policy and takes safer paths.
   - *Off-Policy (Q-Learning)*: Target policy (Greedy $\max_a$) differs from behavior policy ($\varepsilon$-greedy); learns the strictly optimal policy regardless of exploratory noise.
6. **Deep Q-Networks (DQN)**:
   - Uses deep neural networks as nonlinear function approximators for high-dimensional or continuous state spaces.
   - Solves the "Deadly Triad" instability using an **Experience Replay Buffer** (breaks temporal correlations) and a **Separate Target Network** (prevents moving-target oscillations).
