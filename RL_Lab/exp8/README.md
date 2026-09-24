# Experiment 8: SARSA Algorithm (On-Policy TD Control)

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To implement the SARSA ($S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$) on-policy Temporal Difference control algorithm on a standard Reinforcement Learning environment (Cliff Walking), train an agent using $\varepsilon$-greedy exploration, and evaluate the learned policy and path safety characteristics.

---

## 2. Objectives
1. Model the standard $4 \times 12$ Cliff Walking grid environment (Sutton & Barto Example 6.6).
2. Formulate the SARSA update equation based on state-action-reward-state-action transitions.
3. Train the agent using an exponential decaying $\varepsilon$-greedy exploration schedule over 500 episodes.
4. Track online learning metrics: cumulative rewards and steps per episode.
5. Demonstrate the learned "safe path" chosen by SARSA to avoid exploratory cliff falls.

---

## 3. Mathematical Formulations & Theoretical Background

### 3.1 On-Policy TD Control: SARSA
SARSA evaluates and improves the same policy used to make decisions. The algorithm transitions through the sequence:
$$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$$

The action-value update rule is:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

where:
- $\alpha \in (0, 1]$ is the learning rate.
- $\gamma \in [0, 1]$ is the discount factor.
- $A_{t+1}$ is the actual action selected in state $S_{t+1}$ according to the current $\varepsilon$-greedy policy.

### 3.2 The Cliff Walking Problem Dynamics
- Grid Size: $4 \times 12$ ($48$ states).
- Start state: $(3, 0)$ [bottom-left].
- Goal state: $(3, 11)$ [bottom-right].
- The Cliff: States $(3, 1)$ through $(3, 10)$. Stepping into the cliff yields reward $-100$ and teleports the agent back to Start $(3, 0)$.
- Standard step penalty: $-1$ per step.

### 3.3 Why SARSA Learns the "Safe Path"
Because SARSA is **on-policy**, the update for $Q(S_t, A_t)$ includes the term $Q(S_{t+1}, A_{t+1})$, where $A_{t+1}$ might be an exploratory random action with probability $\varepsilon$. Near the edge of the cliff (row 2), an accidental exploratory step DOWN falls into the cliff ($-100$ penalty). 
SARSA anticipates this exploratory risk during training and depresses the value of actions near the cliff edge. Consequently, SARSA converges to the **safer path** (routing along the upper rows), ensuring maximum safety during online exploration.

---

## 4. Algorithm Step-by-Step

```text
Algorithm: SARSA (On-Policy TD Control) for Estimating Q ~ q*
Input: alpha in (0, 1], small epsilon > 0, discount gamma
Initialize Q(s, a) arbitrarily (e.g. 0), and Q(terminal, .) = 0

Loop for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., eps-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (eps-greedy)
        Q(S, A) <- Q(S, A) + alpha * [ R + gamma * Q(S', A') - Q(S, A) ]
        S <- S'
        A <- A'
    Until S is terminal
```

---

## 5. Execution Instructions

Run from repository root:
```bash
python exp8/sarsa.py
```

Generated outputs:
- `exp8/output.txt`: Training progress logs across 500 episodes and evaluation trajectory coordinates.
- `exp8/sarsa_cliffwalking.png`: Dual-panel plot displaying the episode reward learning curve and the 2D grid path trajectory.

---

## 6. Observed Results & Discussion

### Training Progress (500 Episodes):
- **Episode 50**: Avg Reward = $-2498.2$ | Avg Steps = $357.8$ ($\varepsilon = 0.7783$)
- **Episode 150**: Avg Reward = $-134.1$ | Avg Steps = $56.8$ ($\varepsilon = 0.4715$)
- **Episode 300**: Avg Reward = $-25.0$ | Avg Steps = $23.0$ ($\varepsilon = 0.2223$)
- **Episode 500**: Avg Reward = $-22.8$ | Avg Steps = $18.8$ ($\varepsilon = 0.0816$)

### Deterministic Greedy Policy Evaluation ($\varepsilon = 0$):
- **Total Test Steps to Goal**: $17$
- **Total Test Reward**: $-17.0$
- **State Path Traversed**:
  $$(3, 0) \to (2, 0) \to (1, 0) \to (0, 0) \to (0, 1) \to \dots \to (0, 11) \to (1, 11) \to (2, 11) \to (3, 11)$$
- **Actions**:
  $$\text{UP} \to \text{UP} \to \text{UP} \to 11 \times \text{RIGHT} \to \text{DOWN} \to \text{DOWN} \to \text{DOWN}$$

### Path Safety Analysis:
The agent traversed along row 0 (the top-most row), maintaining a safe buffer distance of 3 full cells away from the cliff edge. This confirms SARSA's inherent risk-aversion in environments with exploratory penalties.

---

## 7. Viva-Voce Questions & Answers

**Q1: Why is SARSA called an "On-Policy" TD control algorithm?**  
*Answer*: Because the target policy (the policy whose value function is being estimated) is identical to the behavior policy (the policy being used to select actions and generate transitions). Specifically, SARSA updates $Q(S_t, A_t)$ using the next action $A_{t+1}$ actually sampled by the current policy.

**Q2: What is the significance of the acronym SARSA?**  
*Answer*: SARSA stands for **S**tate, **A**ction, **R**eward, **S**tate, **A**ction—the exact 5-tuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ needed to execute one transition update.

**Q3: How does SARSA differ from Q-Learning in its update rule?**  
*Answer*:
- **SARSA**: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$ (uses actual action $A_{t+1}$).
- **Q-Learning**: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$ (uses max over all possible actions).

**Q4: Under what conditions is SARSA guaranteed to converge to the optimal policy?**  
*Answer*: SARSA converges to $q^*$ with probability 1 if:
1. All state-action pairs are visited infinitely often (guaranteed by $\varepsilon$-soft policies).
2. The policy converges to the greedy policy in the limit (e.g. $\varepsilon \to 0$ as $t \to \infty$).
3. The step sizes satisfy standard Robbins-Monro conditions: $\sum_{t=1}^\infty \alpha_t = \infty$ and $\sum_{t=1}^\infty \alpha_t^2 < \infty$.
