# Experiment 4: Policy Iteration

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To implement Generalized Policy Iteration (GPI) consisting of policy evaluation and policy improvement steps to compute the optimal state-value function $V^*(s)$ and optimal deterministic policy $\pi^*(s)$ for a Grid World problem containing obstacles and hazards.

---

## 2. Objectives
1. Model an enhanced $4 \times 4$ Grid World with target Goal $(3, 3)$ ($+10.0$), Trap $(1, 2)$ ($-10.0$), and an impassable Wall/Obstacle at $(1, 1)$.
2. Implement Policy Evaluation: iteratively compute $V^\pi(s)$ until $\max_s |V_{k+1}(s) - V_k(s)| < \theta$.
3. Implement Policy Improvement: update $\pi(s) \leftarrow \arg\max_a q_\pi(s, a)$.
4. Detect policy stability ($\pi_{k+1} = \pi_k$) to terminate outer iteration.
5. Visualize the resulting optimal policy vectors and optimal state values.

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 Generalized Policy Iteration (GPI)
GPI refers to the general concept of allowing policy evaluation and policy improvement processes to interact:
$$\pi_0 \xrightarrow{E} V^{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} V^{\pi_1} \xrightarrow{I} \dots \xrightarrow{E} V^* \xrightarrow{I} \pi^*$$

### 3.2 Policy Evaluation Update
For a given deterministic policy $\pi$:
$$V_{k+1}(s) = \mathcal{R}(s, \pi(s)) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, \pi(s)) V_k(s')$$

### 3.3 Policy Improvement Theorem
Let $\pi$ and $\pi'$ be any pair of deterministic policies such that for all $s \in \mathcal{S}$:
$$q_\pi(s, \pi'(s)) \ge V^\pi(s)$$
Then policy $\pi'$ must be as good as, or better than, policy $\pi$:
$$V^{\pi'}(s) \ge V^\pi(s) \quad \forall s \in \mathcal{S}$$

The greedy policy update rule is:
$$\pi'(s) \doteq \arg\max_{a \in \mathcal{A}} q_\pi(s, a) = \arg\max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]$$

When $\pi'(s) = \pi(s)$ for all $s$, both the policy and the value function satisfy the Bellman Optimality Equation, proving that $\pi = \pi^*$ and $V = V^*$.

---

## 4. Algorithm Step-by-Step

```text
Algorithm: Policy Iteration for Optimal Policy pi*
Input: Environment with S, A, P, R, discount factor gamma, threshold theta
Initialize V(s) = 0, arbitrary policy pi(s) in A for all s in S

Loop (Outer GPI Cycle):
    # 1. Policy Evaluation
    Loop:
        Delta <- 0
        For each non-terminal, non-obstacle s in S:
            v <- V(s)
            V(s) <- R(s, pi(s)) + gamma * sum_{s'} P(s'|s, pi(s)) * V(s')
            Delta <- max(Delta, |v - V(s)|)
        Until Delta < theta

    # 2. Policy Improvement
    policy_stable <- True
    For each non-terminal, non-obstacle s in S:
        old_action <- pi(s)
        pi(s) <- argmax_a [ R(s, a) + gamma * sum_{s'} P(s'|s, a) * V(s') ]
        If old_action != pi(s):
            policy_stable <- False

    If policy_stable:
        Stop and Return V* = V, pi* = pi
```

---

## 5. Execution Instructions

Run from repository root:
```bash
python exp4/policy_iteration.py
```

Generated outputs:
- `exp4/output.txt`: Convergence trace of evaluation sweeps, optimal values, and directional actions.
- `exp4/policy_iteration_result.png`: Heatmap with overlaid policy directions and cell annotations.

---

## 6. Observed Results & Discussion

### Converged Results Summary
- **Policy Improvement Cycles**: 5
- **Total Policy Evaluation Sweeps**: 282 (Initial evaluation took 271 sweeps, subsequent evaluations took 3, 3, 4, 1 sweeps due to warm-starting).

### Optimal State-Value Matrix $V^*(s)$:
$$\begin{bmatrix}
3.21 & 4.44 & 5.72 & 7.07 \\
4.44 & \text{WALL} & \text{TRAP} & 8.50 \\
5.72 & 7.07 & 8.50 & 10.00 \\
7.07 & 8.50 & 10.00 & \text{GOAL}
\end{bmatrix}$$

### Optimal Directional Policy $\pi^*(s)$:
$$\begin{bmatrix}
\downarrow \text{ DOWN} & \rightarrow \text{ RIGHT} & \rightarrow \text{ RIGHT} & \downarrow \text{ DOWN} \\
\downarrow \text{ DOWN} & \text{WALL} & \text{TRAP} & \downarrow \text{ DOWN} \\
\downarrow \text{ DOWN} & \downarrow \text{ DOWN} & \downarrow \text{ DOWN} & \downarrow \text{ DOWN} \\
\rightarrow \text{ RIGHT} & \rightarrow \text{ RIGHT} & \rightarrow \text{ RIGHT} & \text{GOAL}
\end{bmatrix}$$

### Safety & Obstacle Avoidance:
- States adjacent to the Trap $(1, 2)$ steer away from it: cell $(0, 2)$ moves RIGHT to $(0, 3)$ and then DOWN, rather than stepping down into the trap.
- Cells bordering the Wall $(1, 1)$ correctly route around it toward the high-value Goal $(3, 3)$.

---

## 7. Viva-Voce Questions & Answers

**Q1: What guarantees that Policy Iteration will converge in a finite number of iterations?**  
*Answer*: For any finite MDP, the number of possible deterministic policies is finite ($|\mathcal{A}|^{|\mathcal{S}|}$). By the Policy Improvement Theorem, each strictly improved policy has a strictly greater value function ($V^{\pi_{k+1}} > V^{\pi_k}$). Because no policy can be visited twice and the total number of policies is finite, Policy Iteration must terminate in a finite number of iterations.

**Q2: What is the main computational bottleneck in Policy Iteration?**  
*Answer*: The Policy Evaluation step. Computing $V^\pi(s)$ to near-convergence at every outer cycle requires many iterative sweeps through the state space. This motivation leads directly to Value Iteration, which truncates evaluation to a single sweep.

**Q3: Can we initialize $V(s)$ with values from the previous policy evaluation step?**  
*Answer*: Yes, this is known as "warm-starting". Because $\pi_{k+1}$ is typically similar to $\pi_k$, $V^{\pi_k}$ serves as an excellent initial estimate for $V^{\pi_{k+1}}$, drastically reducing the evaluation sweeps required in subsequent cycles (as seen in our run: from 271 sweeps down to 3 sweeps).

**Q4: How does Policy Iteration handle stochastic transitions?**  
*Answer*: The policy evaluation and greedy improvement equations compute expectations across all probable successor states weighted by $P(s' \mid s, a)$, naturally accounting for stochasticity.
