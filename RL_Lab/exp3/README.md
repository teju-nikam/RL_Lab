# Experiment 3: Dynamic Programming – Policy Evaluation

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To implement and analyze the Iterative Policy Evaluation algorithm using Dynamic Programming for an equiprobable random policy in a standard $4 \times 4$ Grid World environment.

---

## 2. Objectives
1. Model the classic Sutton & Barto $4 \times 4$ Grid World benchmark with terminal states at $(0, 0)$ and $(3, 3)$.
2. Implement the synchronous/iterative Bellman Expectation update rule.
3. Track value function convergence across intermediate sweeps ($k = 0, 1, 2, 10,$ and converged state).
4. Verify convergence criteria using a threshold $\theta = 10^{-5}$.
5. Generate heatmaps visualizing the spatial decay of expected discounted returns.

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 Policy Evaluation Problem
Given a policy $\pi$, policy evaluation computes the state-value function $V^\pi(s)$ for all $s \in \mathcal{S}$:
$$V^\pi(s) \doteq \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right]$$

### 3.2 Bellman Expectation Equation for $V^\pi(s)$
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V^\pi(s') \right]$$

In deterministic Grid World transitions with step cost $r = -1$:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ -1 + \gamma V^\pi(s') \right]$$

### 3.3 Iterative Update Rule (Dynamic Programming)
Starting with arbitrary initial values $V_0(s) = 0$, successive approximations $V_{k+1}$ are computed from $V_k$ using the update:
$$V_{k+1}(s) \leftarrow \sum_{a} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V_k(s') \right]$$

Convergence is guaranteed by Banach's Fixed-Point Theorem because the Bellman Expectation operator is a $\gamma$-contraction mapping under the supremum norm.

---

## 4. Algorithm Step-by-Step

```text
Algorithm: Iterative Policy Evaluation
Input: Environment (S, A, P, R), Policy pi(a|s), discount factor gamma, threshold theta
Initialize V(s) = 0 for all s in S; V(terminal) = 0

Loop:
    Delta <- 0
    For each s in S:
        If s is terminal: continue
        v <- V(s)
        V_new(s) <- sum_{a} pi(a|s) * [ R(s, a) + gamma * sum_{s'} P(s'|s, a) * V(s') ]
        Delta <- max(Delta, |v - V_new(s)|)
    V <- V_new
Until Delta < theta

Return V
```

---

## 5. Execution Instructions

Run from repository root:
```bash
python exp3/policy_evaluation.py
```

Generated outputs:
- `exp3/output.txt`: Numerical state value matrices at sweeps $k = 0, 1, 2, 10$ and convergence.
- `exp3/policy_evaluation_grid.png`: 5-panel heatmap visualization showing value function evolution.

---

## 6. Observed Results & Discussion

### Converged Value Function Matrix (k = 215 sweeps, $\theta = 10^{-5}$)

$$\begin{bmatrix}
0.0 & -14.0 & -20.0 & -22.0 \\
-14.0 & -18.0 & -20.0 & -20.0 \\
-20.0 & -20.0 & -18.0 & -14.0 \\
-22.0 & -20.0 & -14.0 & 0.0
\end{bmatrix}$$

### Key Insights:
1. **Exact Reproduction**: The converged values match Sutton & Barto Chapter 4 (Figure 4.1) precisely.
2. **Symmetry**: Symmetrical states relative to the two terminal corners (e.g. $(0, 1)$ and $(1, 0)$) have identical values of $-14.0$.
3. **Worst States**: The cells furthest from both exits—$(0, 3)$ and $(3, 0)$—have the worst expected return of $-22.0$, reflecting longer random walks before absorption.

---

## 7. Viva-Voce Questions & Answers

**Q1: What is the difference between In-Place and Synchronous (Two-Array) Policy Evaluation?**  
*Answer*: Synchronous evaluation maintains two arrays ($V_{old}$ and $V_{new}$), updating all states based strictly on values from the previous sweep. In-place evaluation uses a single array and immediately uses newly updated state values within the same sweep, which accelerates convergence and requires half the memory.

**Q2: Why does Iterative Policy Evaluation require complete knowledge of the environment model?**  
*Answer*: The Bellman update requires explicit transition probabilities $\mathcal{P}(s' \mid s, a)$ and reward distributions $\mathcal{R}(s, a)$ to compute the expectation over all possible next states. This distinguishes Dynamic Programming (model-based) from Monte Carlo and Temporal Difference learning (model-free).

**Q3: How does the convergence threshold $\theta$ affect the result?**  
*Answer*: $\theta$ specifies the maximum allowable absolute change across all state values ($\max_s |V_{k+1}(s) - V_k(s)| < \theta$). A smaller $\theta$ yields higher numerical precision but requires more sweeps.

**Q4: Under what condition is convergence guaranteed when $\gamma = 1.0$?**  
*Answer*: When $\gamma = 1.0$ (undiscounted), convergence is guaranteed provided all states eventually transition into an absorbing terminal state with probability 1 under policy $\pi$ (episodic task).
