# Experiment 5: Value Iteration & Comparison with Policy Iteration

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To implement the Value Iteration dynamic programming algorithm, extract the optimal deterministic policy $\pi^*(s)$ for a Grid World environment, and conduct a comparative benchmark against Policy Iteration in terms of convergence speed, sweep counts, computational complexity, and policy quality.

---

## 2. Objectives
1. Implement the Bellman Optimality update operator for Value Iteration:
   $$V_{k+1}(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V_k(s') \right]$$
2. Terminate value iteration when the maximum value change falls below a threshold $\theta = 10^{-6}$.
3. Greedily extract the optimal deterministic policy $\pi^*(s)$.
4. Directly compare Value Iteration against Policy Iteration (from Experiment 4) under identical environment dynamics.
5. Demonstrate mathematical equivalence of the final converged value functions ($V^*_{VI} \equiv V^*_{PI}$).

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 Bellman Optimality Equation
The unique optimal state-value function $V^*(s)$ satisfies:
$$V^*(s) = \max_{a \in \mathcal{A}} \mathbb{E} \left[ R_{t+1} + \gamma V^*(S_{t+1}) \;\middle|\; S_t = s, A_t = a \right]$$
$$V^*(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^*(s') \right]$$

### 3.2 Value Iteration Update Rule
Unlike Policy Iteration (which alternates between complete policy evaluation and policy improvement), Value Iteration turns the Bellman Optimality Equation into an update rule:
$$V_{k+1}(s) \leftarrow \max_{a \in \mathcal{A}} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_k(s') \right]$$

This combines policy evaluation and policy improvement into a single sweep per iteration.

### 3.3 Contraction Mapping Theorem
The Bellman optimality operator $T^*$ defined by:
$$(T^* V)(s) \doteq \max_a \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s') \right]$$
is a $\gamma$-contraction in the supremum norm:
$$\|T^* U - T^* V\|_\infty \le \gamma \|U - V\|_\infty$$
By Banach's fixed-point theorem, repeated application of $T^*$ converges to a unique fixed point $V^*$ from any arbitrary initial vector $V_0$.

---

## 4. Algorithm Step-by-Step

```text
Algorithm: Value Iteration
Input: Environment (S, A, P, R), discount factor gamma, threshold theta
Initialize V(s) = 0 for all s in S; V(terminal) = 0

Loop:
    Delta <- 0
    For each non-terminal, non-obstacle s in S:
        v <- V(s)
        V(s) <- max_a [ R(s, a) + gamma * sum_{s'} P(s'|s, a) * V(s') ]
        Delta <- max(Delta, |v - V(s)|)
Until Delta < theta

Output deterministic policy:
pi*(s) = argmax_a [ R(s, a) + gamma * sum_{s'} P(s'|s, a) * V(s') ]
```

---

## 5. Execution Instructions

Run from repository root:
```bash
python exp5/value_iteration.py
```

Generated outputs:
- `exp5/output.txt`: Comparative performance table, sweep counts, runtime, and final values.
- `exp5/value_vs_policy_iteration.png`: Dual-panel comparative visualization of convergence errors and runtime efficiency.

---

## 6. Comparative Benchmark Results

| Metric | Value Iteration | Policy Iteration | Comparison & Analysis |
| :--- | :---: | :---: | :--- |
| **Total State Sweeps** | **7** | **282** (outer cycles: 5) | VI converges in a fraction of sweeps |
| **Execution Time** | **0.380 ms** | **5.000 ms** | VI is ~13x faster |
| **Max Absolute Difference $|V_{VI}^* - V_{PI}^*|$** | **0.00e+00** | **0.00e+00** | **Mathematically Identical** |
| **Policy Equivalence** | **100% Match** | **100% Match** | Both find identical optimal paths |

### Optimal State Values $V^*(s)$:
$$\begin{bmatrix}
3.21 & 4.44 & 5.72 & 7.07 \\
4.44 & \text{WALL} & \text{TRAP} & 8.50 \\
5.72 & 7.07 & 8.50 & 10.00 \\
7.07 & 8.50 & 10.00 & \text{GOAL}
\end{bmatrix}$$

---

## 7. Viva-Voce Questions & Answers

**Q1: What is the fundamental operational difference between Value Iteration and Policy Iteration?**  
*Answer*: Policy Iteration maintains an explicit policy $\pi$, evaluates it to full convergence ($V^\pi$), and then performs a policy improvement step. Value Iteration operates purely in the value space by applying the $\max_a$ operator at every sweep, effectively performing one sweep of policy evaluation followed by immediate greedy improvement without maintaining an explicit intermediate policy.

**Q2: When would Policy Iteration be preferred over Value Iteration?**  
*Answer*: Policy Iteration often converges in fewer outer iterations (policies usually stabilize long before the value function converges to high numerical precision). When state spaces are small and linear systems can be solved directly via matrix inversion, Policy Iteration can terminate very quickly.

**Q3: Why is Value Iteration often faster in practice on discrete Grid Worlds?**  
*Answer*: Value Iteration avoids the excessive computation of fully evaluating suboptimal intermediate policies. In our Grid World benchmark, Value Iteration found the optimal values in just 7 sweeps compared to 282 sweeps for Policy Iteration.

**Q4: How can the optimal policy be extracted once Value Iteration terminates?**  
*Answer*: By conducting a one-step greedy lookahead:
$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^*(s') \right]$$
No model-free action-value table is required if the transition dynamics $\mathcal{P}$ and $\mathcal{R}$ are known.
