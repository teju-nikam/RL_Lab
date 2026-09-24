# Experiment 7: Temporal Difference (TD) Learning

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To implement the $TD(0)$ prediction algorithm using one-step bootstrapping updates on the classic 5-State Random Walk Markov Reward Process, evaluate value estimation accuracy against true analytical values, and compare performance with Constant-$\alpha$ Monte Carlo prediction across multiple learning rates.

---

## 2. Objectives
1. Implement the 5-State Random Walk Markov Reward Process with states $\{A, B, C, D, E\}$ and terminal states $\{L, R\}$.
2. Derive analytical ground-truth state values under an equiprobable random walk policy.
3. Implement the online one-step bootstrapping $TD(0)$ update rule:
   $$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$
4. Implement Constant-$\alpha$ Monte Carlo prediction using complete episodic returns $G_t$.
5. Perform an empirical evaluation comparing Root Mean Square Error (RMSE) against true state values over 100 independent runs.

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 Random Walk Environment Dynamics
- Non-terminal states: $S \in \{A, B, C, D, E\}$ with start state $S_0 = C$.
- Absorbing boundaries: Left terminal state with reward $0$; Right terminal state with reward $+1$.
- Transition probabilities: $P(S_{t+1} = s-1 \mid S_t = s) = 0.5$, $P(S_{t+1} = s+1 \mid S_t = s) = 0.5$.
- Discount factor: $\gamma = 1.0$.

#### Analytical Solution:
By setting up the Bellman expectation equations:
$$V(s) = 0.5 \cdot V(s-1) + 0.5 \cdot V(s+1)$$
with boundary conditions $V(\text{Left}) = 0$ and $V(\text{Right}) = 1$, the analytical solution is linear:
$$V^*(A) = \frac{1}{6} \approx 0.1667, \quad V^*(B) = \frac{2}{6} \approx 0.3333, \quad V^*(C) = \frac{3}{6} = 0.5000$$
$$V^*(D) = \frac{4}{6} \approx 0.6667, \quad V^*(E) = \frac{5}{6} \approx 0.8333$$

### 3.2 TD(0) Prediction Update
Temporal Difference learning combines the sampling of Monte Carlo with the bootstrapping of Dynamic Programming:
$$V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t$$
where $\delta_t$ is the one-step **TD Error**:
$$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

The target is $R_{t+1} + \gamma V(S_{t+1})$ (an estimate of the true return).

### 3.3 Constant-$\alpha$ Monte Carlo Update
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$
where $G_t$ is the actual accumulated discounted return from time $t$ until the end of the episode.

### 3.4 Root Mean Square Error (RMSE) Metric
$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( V(s) - V_{\text{true}}(s) \right)^2 }$$

---

## 4. Algorithm Step-by-Step

```text
Algorithm: Tabular TD(0) for estimating V^pi
Input: Policy pi to evaluate, step size alpha in (0, 1], discount factor gamma
Initialize V(s) arbitrarily (e.g. 0.5 for all s in S), V(terminal) = 0

Loop for each episode:
    Initialize S
    Loop for each step of episode:
        A <- action given by pi for S
        Take action A, observe R, S'
        V(S) <- V(S) + alpha * [ R + gamma * V(S') - V(S) ]
        S <- S'
    Until S is terminal
```

---

## 5. Execution Instructions

Run from repository root:
```bash
python exp7/td_learning.py
```

Generated outputs:
- `exp7/output.txt`: Value estimates across episodes and empirical RMSE table over 100 runs.
- `exp7/td_vs_mc_comparison.png`: Dual-panel plot comparing value progression and learning curves (RMSE vs episodes).

---

## 6. Observed Results & Discussion

### Value Function Progression ($\alpha = 0.1$):
| State | True Value | 0 Ep | 1 Ep | 10 Ep | 25 Ep | 100 Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **0.1667** | 0.5000 | 0.5000 | 0.3555 | 0.2249 | **0.1323** |
| **B** | **0.3333** | 0.5000 | 0.5000 | 0.4625 | 0.3742 | **0.3870** |
| **C** | **0.5000** | 0.5000 | 0.5000 | 0.5169 | 0.4738 | **0.5713** |
| **D** | **0.6667** | 0.5000 | 0.5000 | 0.5911 | 0.5823 | **0.7354** |
| **E** | **0.8333** | 0.5000 | 0.5500 | 0.6704 | 0.7660 | **0.8248** |

### Empirical RMSE Benchmark (100 Runs, 100 Episodes):
| Algorithm | Learning Rate ($\alpha$) | Final RMSE | Performance Rank |
| :--- | :---: | :---: | :---: |
| **TD(0)** | **0.05** | **0.0356** | **1 (Best Accuracy)** |
| **TD(0)** | **0.10** | **0.0597** | **2** |
| **Monte Carlo** | **0.03** | **0.0615** | **3** |
| **Monte Carlo** | **0.04** | **0.0619** | **4** |
| **TD(0)** | **0.15** | **0.0767** | **5** |
| **Monte Carlo** | **0.02** | **0.0781** | **6** |
| **Monte Carlo** | **0.01** | **0.1280** | **7** |

### Key Takeaways:
1. **Bootstrapping Advantage**: TD(0) updates immediately after every single step using the estimated successor value $V(S_{t+1})$. This drastically reduces variance compared to Monte Carlo.
2. **Speed of Convergence**: TD learns significantly faster on Markov tasks because it exploits the Markov property, whereas MC updates ignore transitions and treat episodes as independent trajectories.

---

## 7. Viva-Voce Questions & Answers

**Q1: What is bootstrapping in reinforcement learning?**  
*Answer*: Bootstrapping refers to updating an estimate of a value function based on other existing estimates, rather than waiting for actual terminal rewards. Dynamic Programming and Temporal Difference learning bootstrap; Monte Carlo does not bootstrap.

**Q2: What is the bias-variance tradeoff between TD(0) and Monte Carlo?**  
*Answer*:
- **Monte Carlo**: Zero bias (target $G_t$ is an unbiased estimate of $V^\pi(S_t)$), but **high variance** because $G_t$ depends on many random actions and state transitions.
- **TD(0)**: Some bias (because target $R + \gamma V(S')$ relies on an imperfect current estimate $V$), but **low variance** because it depends only on a single transition.

**Q3: Can TD(0) be applied to continuous / non-terminating tasks?**  
*Answer*: Yes! Because TD(0) updates after every transition ($t \to t+1$), it does not require an episode to terminate. Monte Carlo, on the other hand, strictly requires episodic termination to compute $G_t$.

**Q4: What is the TD error $\delta_t$, and what does it represent?**  
*Answer*: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$. It represents the surprise or unexpected difference between the newly observed one-step estimate ($R_{t+1} + \gamma V(S_{t+1})$) and the previous expectation ($V(S_t)$).
