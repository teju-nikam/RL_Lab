# Experiment 1: Multi-Armed Bandit Problem

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To implement the Multi-Armed Bandit testbed, evaluate $\varepsilon$-greedy action selection mechanisms, and compare exploration versus exploitation strategies (Pure Greedy, $\varepsilon$-Greedy, Decaying $\varepsilon$, and Upper Confidence Bound (UCB)).

---

## 2. Objectives
1. Formulate a 10-armed bandit environment with stochastic Gaussian reward distributions.
2. Implement incremental sample-average action-value estimation.
3. Compare action selection strategies:
   - Pure Greedy ($\varepsilon = 0$)
   - Small exploration ($\varepsilon = 0.01$)
   - Moderate exploration ($\varepsilon = 0.10$)
   - Decaying exploration ($\varepsilon_t = \frac{\varepsilon_0}{1 + \lambda t}$)
   - Upper Confidence Bound (UCB1) action selection.
4. Quantify performance across repeated independent trials (1000 runs, 2000 steps per run).
5. Analyze the exploration-exploitation trade-off.

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 The $k$-Armed Bandit Problem
In an action-state problem with only one state (non-associative setting), the agent selects from $k$ discrete actions $a \in \{1, 2, \dots, k\}$. Each action yields a reward drawn from a stationary probability distribution with expected value:
$$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$$

### 3.2 Action-Value Estimation
The estimated value of action $a$ at time step $t$, denoted $Q_t(a)$, is computed using the sample-average method:
$$Q_t(a) \doteq \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}(A_i = a)}{\sum_{i=1}^{t-1} \mathbb{I}(A_i = a)}$$

### 3.3 Incremental Implementation
To avoid storing all historical rewards, the update is computed iteratively in $O(1)$ memory:
$$Q_{n+1} = Q_n + \frac{1}{n} \left[ R_n - Q_n \right]$$
$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \times \left[ \text{Target} - \text{OldEstimate} \right]$$

### 3.4 Action Selection Strategies

1. **Greedy Action Selection ($\varepsilon = 0$)**:
   $$A_t \doteq \arg\max_a Q_t(a)$$
   *Drawback*: Gets trapped in suboptimal actions due to lack of exploration.

2. **$\varepsilon$-Greedy Action Selection**:
   $$A_t = \begin{cases} \arg\max_a Q_t(a) & \text{with probability } 1 - \varepsilon \\ \text{uniform random action } a \in \{1, \dots, k\} & \text{with probability } \varepsilon \end{cases}$$

3. **Decaying $\varepsilon$-Greedy**:
   $$\varepsilon_t = \frac{\varepsilon_0}{1 + \lambda \cdot t}$$
   Allows high initial exploration that smoothly transitions to exploitation.

4. **Upper Confidence Bound (UCB1)**:
   Balances exploitation and exploration based on uncertainty (variance reduction):
   $$A_t \doteq \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
   where $N_t(a)$ is the number of times action $a$ was selected prior to time $t$, and $c > 0$ controls the degree of exploration.

---

## 4. Algorithm Step-by-Step

```text
Algorithm: Multi-Armed Bandit Benchmark
Input: Number of arms k = 10, runs = 1000, steps = 2000, strategies
Initialize arrays: avg_rewards[strategy, step], pct_optimal[strategy, step]

For run = 1 to 1000:
    Initialize Bandit with true values q*(a) ~ N(0, 1)
    For each strategy:
        Initialize Q(a) = 0, N(a) = 0 for all a in {1..k}
        For step = 1 to 2000:
            Select action A based on strategy rule:
                - Greedy / eps-Greedy / Decaying eps / UCB1
            Execute action A, receive reward R ~ N(q*(A), 1)
            N(A) <- N(A) + 1
            Q(A) <- Q(A) + (1 / N(A)) * (R - Q(A))
            Record reward and whether A == argmax(q*)

Compute mean reward and optimal action percentage across all 1000 runs
Plot and log results
```

---

## 5. Execution Instructions

Run the script from the repository root:
```bash
python exp1/bandit.py
```

Generated outputs:
- `exp1/output.txt`: Console logs and final numerical summary table.
- `exp1/bandit_comparison.png`: Dual-panel comparative learning curves.

---

## 6. Observed Results & Discussion

### Final Performance Summary (Averaged over 1000 runs)

| Strategy | Average Reward (Final 100 steps) | % Optimal Action (Final 100 steps) | Rank |
| :--- | :---: | :---: | :---: |
| **UCB ($c=2.0$)** | **~1.45 - 1.55** | **~80% - 85%** | **1** |
| **Decaying $\varepsilon$ ($\varepsilon_0=0.20$)** | **~1.40 - 1.50** | **~80% - 84%** | **2** |
| **$\varepsilon$-Greedy ($\varepsilon=0.01$)** | **~1.35 - 1.45** | **~75% - 80%** | **3** |
| **$\varepsilon$-Greedy ($\varepsilon=0.10$)** | **~1.30 - 1.38** | **~81% - 83%** | **4** |
| **Pure Greedy ($\varepsilon=0.0$)** | **~1.00 - 1.10** | **~35% - 40%** | **5** |

### Key Observations:
1. **Greedy Vulnerability**: Pure greedy action selection plateaus very early around ~35-40% optimal action rate. If the first sampled arm returns a modestly positive reward, the greedy agent locks into it without testing potentially superior alternatives.
2. **Impact of $\varepsilon$**: $\varepsilon = 0.10$ discovers the optimal arm quickly but suffers an asymptotic ceiling of $\approx 91\%$ optimal actions because it intentionally behaves randomly 10% of the time. Conversely, $\varepsilon = 0.01$ climbs slowly but achieves high asymptotic reward.
3. **Superiority of UCB**: UCB intelligently directs exploration towards actions that are either estimated to be promising or have been sampled infrequently, eliminating wasted random exploratory actions.

---

## 7. Viva-Voce Questions & Answers

**Q1: What is the exploration-exploitation dilemma in reinforcement learning?**  
*Answer*: Exploration involves gathering information about uncharted or uncertain actions to discover better options, whereas exploitation involves using current knowledge to pick the action known to yield the highest immediate reward. Over-exploring degrades immediate returns; over-exploiting risks being permanently stuck in suboptimal behaviors.

**Q2: Why is the incremental update formula $Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]$ preferred over summing all past rewards?**  
*Answer*: It requires $O(1)$ constant memory and constant computational time per step, eliminating the need to store historical sequences of rewards $R_1, R_2, \dots, R_n$.

**Q3: How does the UCB algorithm balance exploration and exploitation?**  
*Answer*: UCB uses the term $c \sqrt{\frac{\ln t}{N_t(a)}}$ as an uncertainty bonus. When an action $a$ is selected frequently, $N_t(a)$ increases, shrinking the bonus. As time $t$ progresses without selecting action $a$, $\ln t$ grows, increasing the bonus and eventually forcing the agent to re-explore action $a$.

**Q4: How does stationary bandit differ from non-stationary bandit problems?**  
*Answer*: In stationary bandits, true action values $q_*(a)$ remain constant over time, making sample averages ($\frac{1}{n}$) appropriate. In non-stationary bandits, $q_*(a)$ changes over time, requiring constant step-size parameters $\alpha \in (0, 1]$ (exponential recency-weighted averages) so recent rewards have greater weight than older rewards.
