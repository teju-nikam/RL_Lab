# Experiment 6: Monte Carlo Prediction and Control

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To estimate state-value functions using First-Visit and Every-Visit Monte Carlo Prediction methods, and to learn an optimal policy through model-free episodic interaction using On-Policy First-Visit Monte Carlo Control with $\varepsilon$-greedy exploration.

---

## 2. Objectives
1. Implement model-free episodic learning in a stochastic $4 \times 4$ Grid World with Goal ($+10.0$) and Pit ($-10.0$).
2. Formulate and contrast **First-Visit Monte Carlo** vs. **Every-Visit Monte Carlo** prediction.
3. Implement **On-Policy First-Visit Monte Carlo Control** using an action-value table $Q(s, a)$ and decaying $\varepsilon$-greedy action selection.
4. Verify policy improvement and convergence across 10,000 learning episodes.
5. Generate graphical visualizations of estimated value functions, learning curves, and learned directional policies.

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 Model-Free Monte Carlo Concept
Unlike Dynamic Programming (which requires full transition models $\mathcal{P}(s' \mid s, a)$ and rewards $\mathcal{R}(s, a)$), Monte Carlo methods learn purely from sample sequences of states, actions, and rewards from actual or simulated episodic interaction:
$$S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T$$

The total discounted return following time step $t$ is:
$$G_t \doteq \sum_{k=0}^{T - t - 1} \gamma^k R_{t+k+1}$$

### 3.2 First-Visit vs. Every-Visit MC Prediction
- **First-Visit MC**: Averages returns only from the *first time* state $s$ is visited within an episode:
  $$V(s) = \frac{1}{|Returns(s)|} \sum_{G \in Returns(s)} G$$
  *Property*: Provides an unbiased estimator of $V^\pi(s)$ with variance decreasing as $O(1/\sqrt{N})$.

- **Every-Visit MC**: Averages returns following *every visit* to state $s$ in an episode.
  *Property*: Biased for small samples, but asymptotically unbiased and consistent with lower mean squared error in some continuous formulations.

### 3.3 On-Policy First-Visit Monte Carlo Control
To find an optimal policy without a transition model, we must estimate action-values $Q(s, a)$.
1. **Policy Evaluation**: Following an episode, update action-values incrementally:
   $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} \left[ G_t - Q(S_t, A_t) \right]$$
2. **Policy Improvement**: Update $\varepsilon$-greedy policy derived from $Q$:
   $$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_a Q(s, a) \\ \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a \ne \arg\max_a Q(s, a) \end{cases}$$

---

## 4. Algorithm Step-by-Step

```text
Algorithm: On-Policy First-Visit MC Control (for epsilon-soft policies)
Input: Discount factor gamma, exploration schedule (eps_start, eps_end, eps_decay)
Initialize: Q(s, a) = 0, N(s, a) = 0 for all s in S, a in A

For episode = 1 to num_episodes:
    Generate episode: S_0, A_0, R_1, S_1, A_1, ..., S_{T-1}, A_{T-1}, R_T using eps-greedy(Q)
    G <- 0
    For t = T-1 down to 0:
        G <- gamma * G + R_{t+1}
        Unless pair (S_t, A_t) appeared in S_0, A_0, ..., S_{t-1}, A_{t-1}:
            N(S_t, A_t) <- N(S_t, A_t) + 1
            Q(S_t, A_t) <- Q(S_t, A_t) + (1 / N(S_t, A_t)) * (G - Q(S_t, A_t))
    Decay epsilon

Output pi*(s) = argmax_a Q(s, a)
```

---

## 5. Execution Instructions

Run from repository root:
```bash
python exp6/monte_carlo.py
```

Generated outputs:
- `exp6/output.txt`: Value prediction tables (First-Visit & Every-Visit), training logs across episodes, and learned policy.
- `exp6/mc_prediction_control.png`: 3-panel visualization showing value estimation, learning curve, and optimal policy vectors.

---

## 6. Observed Results & Discussion

### First-Visit vs. Every-Visit Prediction ($V^\pi(s)$ under Random Policy):
- **Max Absolute Difference**: $0.3566$ (showing strong convergence to the true value function).
- Closer states to the Goal exhibit higher values ($\approx -2.5$), while states near the Pit show strong negative values.

### MC Control Convergence:
- **Episode 2000**: Return = $-0.28$ ($\varepsilon = 0.368$)
- **Episode 4000**: Return = $+2.93$ ($\varepsilon = 0.135$)
- **Episode 10000**: Return = $+4.18$ ($\varepsilon = 0.050$)

### Final Learned Directional Policy $\pi^*(s)$:
$$\begin{bmatrix}
\downarrow \text{ DOWN} & \rightarrow \text{ RIGHT} & \rightarrow \text{ RIGHT} & \downarrow \text{ DOWN} \\
\downarrow \text{ DOWN} & \downarrow \text{ DOWN} & \text{PIT} & \downarrow \text{ DOWN} \\
\downarrow \text{ DOWN} & \rightarrow \text{ RIGHT} & \rightarrow \text{ RIGHT} & \downarrow \text{ DOWN} \\
\rightarrow \text{ RIGHT} & \rightarrow \text{ RIGHT} & \rightarrow \text{ RIGHT} & \text{GOAL}
\end{bmatrix}$$

Notice how states $(0, 2)$ and $(2, 2)$ deliberately steer right or down away from the PIT, and all pathways route reliably to the GOAL.

---

## 7. Viva-Voce Questions & Answers

**Q1: What is the defining advantage of Monte Carlo methods over Dynamic Programming?**  
*Answer*: Monte Carlo methods do not require a mathematical model of the environment's transition probabilities $\mathcal{P}(s' \mid s, a)$ or reward functions $\mathcal{R}(s, a)$. They learn directly from raw observed experience trajectories (sample backups).

**Q2: Why are standard Monte Carlo methods restricted to episodic tasks?**  
*Answer*: Monte Carlo updates depend on the actual total discounted return $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$, which can only be computed once an episode reaches an absorbing terminal state $S_T$. In non-terminating tasks, $G_t$ cannot be determined without truncation.

**Q3: What is the purpose of Exploring Starts (ES) in Monte Carlo control, and why did we use $\varepsilon$-greedy exploration instead?**  
*Answer*: Exploring Starts guarantees that all state-action pairs $(s, a)$ are visited infinitely often by specifying that every episode begins at a randomly chosen $(s, a)$ pair. However, in real-world engineering systems, one cannot freely set arbitrary starting states. An $\varepsilon$-greedy policy avoids this impractical assumption by maintaining exploration throughout online execution.

**Q4: Compare the bias and variance of First-Visit vs Every-Visit Monte Carlo.**  
*Answer*: First-visit MC is strictly unbiased ($E[V(s)] = V^\pi(s)$), but has slightly higher variance because it ignores subsequent visits within the same episode. Every-visit MC is biased for small numbers of samples, but is consistent (asymptotically unbiased as $N \to \infty$) and often yields smaller Mean Squared Error (MSE).
