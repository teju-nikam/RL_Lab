# Experiment 2: Markov Decision Process (MDP) Modeling

**Course**: Reinforcement Learning Lab (01AMP403)  
**Department**: CSE-AIML, Final Year B.Tech. (AI) Semester VII  
**Institute**: D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji  

---

## 1. Aim
To formulate a real-world engineering problem as a Markov Decision Process (MDP), formally specify states, actions, transition probabilities, and reward structures, and analytically compute state values using Bellman matrix inversion.

---

## 2. Objectives
1. Formulate an engineering problem: **Autonomous Guided Vehicle (AGV) for Textile Material Transport**.
2. Formulate the mathematical MDP 5-tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.
3. Validate stochasticity conditions of state-action transition probability matrices.
4. Solve the Bellman Expectation Equation analytically using matrix inversion:
   $$V_\pi = (I - \gamma P_\pi)^{-1} R_\pi$$
5. Compare contrasting operational policies (Aggressive vs. Conservative).
6. Simulate episodic trajectories and inspect real-time state progressions.

---

## 3. Engineering Problem Formulation

### System Context
In modern textile manufacturing plants, Autonomous Guided Vehicles (AGVs) navigate between departments to transport raw cotton bales, spun yarn cheeses, and woven fabric rolls while managing electrical energy constraints.

### 3.1 State Space $\mathcal{S}$ ($|\mathcal{S}| = 5$)
- $S_0$: `Charging_Station` – AGV docked at inductive power terminal.
- $S_1$: `Spinning_Dept` – Loading raw cotton or yarn coils.
- $S_2$: `Weaving_Dept` – Delivering yarn packages to automated looms.
- $S_3$: `Quality_Inspection` – Delivering fabric rolls for automated optical inspection.
- $S_4$: `Low_Battery_Warning` – Critical energy depletion mode requiring emergency recovery.

### 3.2 Action Space $\mathcal{A}$ ($|\mathcal{A}| = 4$)
- $A_0$: `Move_Next` – Navigate along floor magnetic guide paths to the next station.
- $A_1$: `Service` – Execute robotic payload loading, unloading, or inspection.
- $A_2$: `Go_Charge` – Divert to charging dock.
- $A_3$: `Wait_Idle` – Stationary standby mode.

### 3.3 Reward Function $\mathcal{R}(s, a)$
- Operational delivery completion: $+25.0$ to $+30.0$
- Material handling / loading: $+15.0$
- Nominal movement power cost: $-1.0$ to $-2.5$
- Charging completion bonus: $+1.0$ to $+5.0$
- Emergency towing / battery fault penalty: $-10.0$ to $-50.0$

### 3.4 Discount Factor $\gamma$
$\gamma = 0.90$, balancing immediate throughput with long-term battery preservation.

---

## 4. Mathematical Derivation: Analytical Bellman Solution

The Bellman Expectation Equation for state values under policy $\pi$ is:
$$V_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_\pi(s') \right]$$

In compact matrix-vector notation:
$$V_\pi = R_\pi + \gamma P_\pi V_\pi$$
$$(I - \gamma P_\pi) V_\pi = R_\pi$$

Since $\gamma < 1$ and $P_\pi$ is a stochastic matrix (spectral radius $\rho(P_\pi) = 1$), $(I - \gamma P_\pi)$ is guaranteed to be strictly diagonally dominant and invertible:
$$V_\pi = (I - \gamma P_\pi)^{-1} R_\pi$$

---

## 5. Execution Instructions

Run the script from the repository root:
```bash
python exp2/mdp_model.py
```

Generated outputs:
- `exp2/output.txt`: Detailed verification logs, analytical values, and trajectory printouts.
- `exp2/mdp_network_and_values.png`: Dual visualization of state values and stationary transition matrix heatmap.

---

## 6. Observed Results & Discussion

### Analytical State Values $V_\pi(s)$

| State | Aggressive Policy $V(s)$ | Conservative Policy $V(s)$ | Engineering Significance |
| :--- | :---: | :---: | :--- |
| **Charging_Station ($S_0$)** | 115.445 | 84.771 | Baseline depot value |
| **Spinning_Dept ($S_1$)** | 130.117 | 95.855 | Inflow loading value |
| **Weaving_Dept ($S_2$)** | 172.284 | 103.323 | High loom throughput reward |
| **Quality_Inspection ($S_3$)** | 177.025 | 95.140 | Final delivery verification |
| **Low_Battery_Warning ($S_4$)** | 85.932 | 59.460 | Emergency recovery penalty state |

### Sample 10-Step Simulation Trajectory:
```text
Step 00: Charging_Station     -> Move_Next  | Reward:  -1.0 | Next: Spinning_Dept
Step 01: Spinning_Dept        -> Service    | Reward: +15.0 | Next: Spinning_Dept
Step 02: Spinning_Dept        -> Move_Next  | Reward:  -2.0 | Next: Low_Battery_Warning
Step 03: Low_Battery_Warning  -> Go_Charge  | Reward: -10.0 | Next: Charging_Station
Step 04: Charging_Station     -> Move_Next  | Reward:  -1.0 | Next: Spinning_Dept
Step 05: Spinning_Dept        -> Service    | Reward: +15.0 | Next: Spinning_Dept
Step 06: Spinning_Dept        -> Service    | Reward: +15.0 | Next: Spinning_Dept
Step 07: Spinning_Dept        -> Service    | Reward: +15.0 | Next: Spinning_Dept
Step 08: Spinning_Dept        -> Service    | Reward: +15.0 | Next: Spinning_Dept
Step 09: Spinning_Dept        -> Service    | Reward: +15.0 | Next: Spinning_Dept
Total Discounted Return (gamma=0.90): 39.206
```

---

## 7. Viva-Voce Questions & Answers

**Q1: What is the Markov Property?**  
*Answer*: The Markov property states that the future is conditionally independent of the past given the present:
$$\mathbb{P}(S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} \mid S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, \dots, S_0 = s_0) = \mathbb{P}(S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} \mid S_t = s_t, A_t = a_t)$$
The current state contains all relevant history needed to decide the future.

**Q2: What are the components of an MDP?**  
*Answer*: An MDP is defined by a 5-tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:
- $\mathcal{S}$: Set of all valid states.
- $\mathcal{A}$: Set of all valid actions.
- $\mathcal{P}$: State transition probability function $P(s' \mid s, a)$.
- $\mathcal{R}$: Reward function $R(s, a, s')$.
- $\gamma \in [0, 1]$: Discount factor determining the present value of future rewards.

**Q3: Why is matrix inversion $(I - \gamma P_\pi)^{-1}$ computationally prohibitive for large MDPs?**  
*Answer*: Inverting an $|\mathcal{S}| \times |\mathcal{S}|$ matrix has a time complexity of $O(|\mathcal{S}|^3)$. While suitable for small problems (e.g., $|\mathcal{S}| \le 1000$), modern RL environments with millions or infinite states require iterative Dynamic Programming, Temporal Difference learning, or function approximation.

**Q4: What is the physical significance of the discount factor $\gamma$?**  
*Answer*: A discount factor $\gamma < 1$ ensures mathematical convergence of infinite-horizon returns ($\sum_{t=0}^\infty \gamma^t R_{t+1} < \infty$). Practically, it models uncertainty about future events, inflation, battery drain, or opportunity cost.
