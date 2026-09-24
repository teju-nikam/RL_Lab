# Experiment 9: Q-Learning Algorithm & Comparison with SARSA

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To implement the tabular Q-Learning (Off-Policy TD Control) algorithm to learn an optimal policy on standard Reinforcement Learning environments (Cliff Walking and FrozenLake-v1), train the agent using $\varepsilon$-greedy exploration, and conduct a detailed performance comparison against SARSA (On-Policy TD Control).

---

## 2. Objectives
1. Implement Q-Learning using the Bellman Optimality update:
   $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
2. Train both Q-Learning and SARSA agents using decaying $\varepsilon$-greedy exploration under identical conditions.
3. Compare the safety vs. optimality tradeoff on the classic Cliff Walking domain:
   - Online training rewards (reward while learning)
   - Evaluated greedy deterministic policy paths (reward after learning)
4. Benchmark both algorithms on the stochastic `FrozenLake-v1` environment and measure test success rates.
5. Generate dual-trajectory plots and reward convergence curves.

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 Q-Learning (Off-Policy TD Control)
In Q-learning, the agent updates its action-value function directly toward the maximum estimated return of the next state, regardless of the action selected by the behavior policy:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

- **Behavior Policy**: $\varepsilon$-greedy policy used to select actions and explore the environment.
- **Target Policy**: Greedy policy $\arg\max_a Q(S, a)$ being evaluated and optimized.
- Because target policy $\ne$ behavior policy, Q-learning is **off-policy**.

### 3.2 Comparison: Q-Learning vs. SARSA Update Rules

| Characteristic | SARSA (On-Policy) | Q-Learning (Off-Policy) |
| :--- | :--- | :--- |
| **Backup Target** | $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ | $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ |
| **Target Policy** | $\varepsilon$-greedy (actual policy being executed) | Greedy $\pi(s) = \arg\max_a Q(s, a)$ |
| **Online Training Behavior** | Learns safe policy that minimizes exploration penalties | Experiences frequent exploratory penalties near hazards |
| **Final Greedy Policy** | Suboptimal/Safe path (e.g. $-17.0$ reward) | Mathematically optimal shortest path ($-13.0$ reward) |

---

## 4. Algorithm Step-by-Step

```text
Algorithm: Q-Learning (Off-Policy TD Control) for Estimating pi ~ pi*
Input: Step size alpha in (0, 1], exploration parameters (eps_start, eps_end, eps_decay)
Initialize Q(s, a) arbitrarily (e.g. 0), Q(terminal, .) = 0

Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A from S using policy derived from Q (e.g. eps-greedy)
        Take action A, observe R, S'
        Q(S, A) <- Q(S, A) + alpha * [ R + gamma * max_a Q(S', a) - Q(S, A) ]
        S <- S'
    Until S is terminal
    Decay epsilon
```

---

## 5. Execution Instructions

Run from repository root:
```bash
python exp9/q_learning.py
```

Generated outputs:
- `exp9/output.txt`: Comparative metrics on Cliff Walking and FrozenLake success rates.
- `exp9/q_learning_vs_sarsa.png`: Dual visual plot showing online learning reward curves and overlaid optimal vs. safe trajectories.

---

## 6. Observed Results & Discussion

### Cliff Walking Benchmark (500 Episodes):

| Performance Metric | Q-Learning (Off-Policy) | SARSA (On-Policy) | Winner / Analysis |
| :--- | :---: | :---: | :--- |
| **Test Reward (Greedy, $\varepsilon=0$)** | **$-13.0$** | **$-17.0$** | **Q-Learning** (Found the strictly optimal minimum-step path) |
| **Test Steps to Goal** | **13 steps** | **17 steps** | **Q-Learning** is 4 steps faster |
| **Online Training Reward (Last 50)** | **$-46.8$** | **$-22.8$** | **SARSA** suffers fewer falls during exploration |

### Trajectory Route Comparison:
- **Q-Learning Path** (Optimal Cliff-Hugging):
  $$(3, 0) \to (2, 0) \xrightarrow{11 \times \text{RIGHT}} (2, 11) \to (3, 11) \quad [\text{Steps: 13}]$$
- **SARSA Path** (Safe Upper Margin):
  $$(3, 0) \to (2, 0) \to (1, 0) \to (0, 0) \xrightarrow{11 \times \text{RIGHT}} (0, 11) \to (1, 11) \to (2, 11) \to (3, 11) \quad [\text{Steps: 17}]$$

### FrozenLake-v1 Stochastic Benchmark (100 Test Episodes):
- **Q-Learning Success Rate**: **77%**
- **SARSA Success Rate**: **69%**

---

## 7. Viva-Voce Questions & Answers

**Q1: What does "off-policy" mean in the context of Q-Learning?**  
*Answer*: Off-policy means that the algorithm learns the value of the optimal greedy policy (the target policy) while acting according to a different exploratory policy (such as $\varepsilon$-greedy behavior policy). This decoupling allows Q-Learning to converge directly to $q^*$ even while continuously exploring.

**Q2: Why does Q-Learning have a worse online training reward than SARSA on Cliff Walking?**  
*Answer*: Q-Learning learns the optimal path right along the cliff edge (row 2). Because the agent acts $\varepsilon$-greedily during training, an exploratory step will occasionally choose DOWN, sending the agent over the cliff ($-100$ penalty). SARSA takes this into account during its updates and chooses the safer path along row 0, yielding higher cumulative online reward during training.

**Q3: How does the maximization in Q-Learning lead to "Maximization Bias"?**  
*Answer*: The target in Q-Learning is $\max_a Q(S_{t+1}, a)$. Because sample estimates have random noise, taking the maximum over noisy estimates produces a positive expectation bias ($\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$). This can cause overoptimistic value estimates, which is addressed by **Double Q-Learning**.

**Q4: Under what conditions is Q-Learning guaranteed to converge to the optimal policy?**  
*Answer*: Q-Learning converges to $q^*$ with probability 1 if:
1. All state-action pairs continue to be visited and updated infinitely often.
2. The step sizes satisfy $\sum_t \alpha_t(s, a) = \infty$ and $\sum_t \alpha_t^2(s, a) < \infty$.
No conditions on the policy itself are required, provided exploration continues.
