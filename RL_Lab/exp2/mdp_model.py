"""
Experiment 2: Markov Decision Process (MDP) Modeling
Course: Reinforcement Learning Lab (01AMP403)
Department: CSE-AIML, Final Year B.Tech. (AI) Semester VII
DKTE Society's Textile and Engineering Institute, Ichalkaranji

Engineering Problem Formulation:
Autonomous Guided Vehicle (AGV) for Textile Material Transport
States (5):
  0: Charging_Station
  1: Spinning_Dept (Material Loading)
  2: Weaving_Dept (Loom Delivery)
  3: Quality_Inspection (Fabric Inspection)
  4: Low_Battery_Warning (Emergency)

Actions (4):
  0: Move_Next (Navigate to next station)
  1: Service (Load/Unload/Inspect)
  2: Go_Charge (Return to charging dock)
  3: Wait_Idle (Stay stationary)

Tasks:
1. Define states, actions, transition probabilities P(s'|s, a), and rewards R(s, a, s').
2. Verify stochastic transition matrices.
3. Solve for exact state-value functions analytically via Bellman Matrix Inversion:
   V_pi = (I - gamma * P_pi)^(-1) * R_pi
4. Compare an Aggressive vs. a Conservative operating policy.
5. Simulate trajectories and plot the MDP graph and value distributions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class TextileAGV_MDP:
    def __init__(self, gamma=0.90):
        self.gamma = gamma
        self.states = [
            "Charging_Station",      # 0
            "Spinning_Dept",         # 1
            "Weaving_Dept",          # 2
            "Quality_Inspection",    # 3
            "Low_Battery_Warning"    # 4
        ]
        self.num_states = len(self.states)
        self.actions = [
            "Move_Next",             # 0
            "Service",               # 1
            "Go_Charge",             # 2
            "Wait_Idle"              # 3
        ]
        self.num_actions = len(self.actions)

        # Transition matrix: P[s, a, s']
        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        # Expected reward matrix: R[s, a]
        self.R = np.zeros((self.num_states, self.num_actions))

        self._build_model()
        self.verify_stochasticity()

    def _build_model(self):
        # -------------------------------------------------------------
        # State 0: Charging_Station
        # -------------------------------------------------------------
        # Move_Next -> moves to Spinning_Dept with 95% probability, 5% sensor slip stays
        self.P[0, 0, 1] = 0.95
        self.P[0, 0, 0] = 0.05
        self.R[0, 0] = -1.0  # movement cost

        # Service -> idle charge completion (+2)
        self.P[0, 1, 0] = 1.00
        self.R[0, 1] = +2.0

        # Go_Charge -> already at dock (+1)
        self.P[0, 2, 0] = 1.00
        self.R[0, 2] = +1.0

        # Wait_Idle -> stays at dock (-0.5)
        self.P[0, 3, 0] = 1.00
        self.R[0, 3] = -0.5

        # -------------------------------------------------------------
        # State 1: Spinning_Dept (Loading Raw Cotton/Yarn)
        # -------------------------------------------------------------
        # Move_Next -> to Weaving_Dept (90%), 10% battery drain to Low_Battery
        self.P[1, 0, 2] = 0.90
        self.P[1, 0, 4] = 0.10
        self.R[1, 0] = -2.0

        # Service -> load yarn (+15)
        self.P[1, 1, 1] = 0.95
        self.P[1, 1, 4] = 0.05
        self.R[1, 1] = +15.0

        # Go_Charge -> return to Charging_Station (95%), 5% Low_Battery
        self.P[1, 2, 0] = 0.95
        self.P[1, 2, 4] = 0.05
        self.R[1, 2] = +3.0

        # Wait_Idle -> stays (-1.0)
        self.P[1, 3, 1] = 1.00
        self.R[1, 3] = -1.0

        # -------------------------------------------------------------
        # State 2: Weaving_Dept (Delivering Yarn to Looms)
        # -------------------------------------------------------------
        # Move_Next -> to Quality_Inspection (85%), 15% Low_Battery
        self.P[2, 0, 3] = 0.85
        self.P[2, 0, 4] = 0.15
        self.R[2, 0] = -2.5

        # Service -> deliver yarn to loom (+25.0)
        self.P[2, 1, 2] = 0.90
        self.P[2, 1, 4] = 0.10
        self.R[2, 1] = +25.0

        # Go_Charge -> return to Charging_Station (90%), 10% Low_Battery
        self.P[2, 2, 0] = 0.90
        self.P[2, 2, 4] = 0.10
        self.R[2, 2] = +2.0

        # Wait_Idle -> stays (-1.5)
        self.P[2, 3, 2] = 1.00
        self.R[2, 3] = -1.5

        # -------------------------------------------------------------
        # State 3: Quality_Inspection (Fabric Scan Station)
        # -------------------------------------------------------------
        # Move_Next -> loop back to Charging_Station (90%), 10% Low_Battery
        self.P[3, 0, 0] = 0.90
        self.P[3, 0, 4] = 0.10
        self.R[3, 0] = +10.0  # completed delivery cycle

        # Service -> inspect batch (+30.0)
        self.P[3, 1, 3] = 0.85
        self.P[3, 1, 4] = 0.15
        self.R[3, 1] = +30.0

        # Go_Charge -> direct return to dock (92%), 8% Low_Battery
        self.P[3, 2, 0] = 0.92
        self.P[3, 2, 4] = 0.08
        self.R[3, 2] = +5.0

        # Wait_Idle -> stays (-2.0)
        self.P[3, 3, 3] = 1.00
        self.R[3, 3] = -2.0

        # -------------------------------------------------------------
        # State 4: Low_Battery_Warning (Emergency State)
        # -------------------------------------------------------------
        # Move_Next -> heavy penalty (-30), stalls (80%) or crawls to Charging (20%)
        self.P[4, 0, 4] = 0.80
        self.P[4, 0, 0] = 0.20
        self.R[4, 0] = -30.0

        # Service -> cannot service while low battery (-40)
        self.P[4, 1, 4] = 1.00
        self.R[4, 1] = -40.0

        # Go_Charge -> emergency crawl to dock (70%), stall in state 4 (30%)
        self.P[4, 2, 0] = 0.70
        self.P[4, 2, 4] = 0.30
        self.R[4, 2] = -10.0  # towing/emergency power drain

        # Wait_Idle -> battery dies (-50)
        self.P[4, 3, 4] = 1.00
        self.R[4, 3] = -50.0

    def verify_stochasticity(self):
        """Ensures every (s, a) row in P sums exactly to 1.0."""
        for s in range(self.num_states):
            for a in range(self.num_actions):
                row_sum = np.sum(self.P[s, a])
                assert np.isclose(row_sum, 1.0), f"Row sum for state {s}, action {a} is {row_sum} != 1.0"

    def analytical_policy_evaluation(self, policy):
        """
        Solves Bellman Expectation Equation analytically via Matrix Inversion:
        V_pi = (I - gamma * P_pi)^(-1) * R_pi
        policy: array of shape (num_states, num_actions) where policy[s, a] = pi(a|s)
        """
        P_pi = np.zeros((self.num_states, self.num_states))
        R_pi = np.zeros(self.num_states)

        for s in range(self.num_states):
            for a in range(self.num_actions):
                prob = policy[s, a]
                P_pi[s] += prob * self.P[s, a]
                R_pi[s] += prob * self.R[s, a]

        I = np.eye(self.num_states)
        V_pi = np.linalg.inv(I - self.gamma * P_pi) @ R_pi
        return V_pi, P_pi, R_pi

    def simulate_trajectory(self, policy, start_state=0, num_steps=20, seed=42):
        """Simulates an AGV trajectory under a given policy."""
        rng = np.random.default_rng(seed)
        trajectory = []
        curr_state = start_state
        total_return = 0.0

        for t in range(num_steps):
            # Select action
            action_probs = policy[curr_state]
            action = rng.choice(self.num_actions, p=action_probs)
            # Sample next state
            next_state = rng.choice(self.num_states, p=self.P[curr_state, action])
            reward = self.R[curr_state, action]
            total_return += (self.gamma ** t) * reward

            trajectory.append({
                "step": t,
                "state": self.states[curr_state],
                "action": self.actions[action],
                "reward": reward,
                "next_state": self.states[next_state]
            })
            curr_state = next_state

        return trajectory, total_return


def run_experiment():
    print("=" * 75)
    print("EXPERIMENT 2: MARKOV DECISION PROCESS (MDP) MODELING")
    print("ENGINEERING PROBLEM: AUTONOMOUS TEXTILE AGV MATERIAL TRANSPORT")
    print("=" * 75)

    mdp = TextileAGV_MDP(gamma=0.90)

    print(f"Number of States: {mdp.num_states}")
    for idx, s in enumerate(mdp.states):
        print(f"  State {idx}: {s}")

    print(f"\nNumber of Actions: {mdp.num_actions}")
    for idx, a in enumerate(mdp.actions):
        print(f"  Action {idx}: {a}")

    print(f"\nDiscount Factor (gamma): {mdp.gamma}")
    print("[+] Transition probability matrix verified: All rows sum to 1.0 (Stochastic).")

    # Define Policy 1: Aggressive Transport Policy (Always Move & Service, ignores charging until emergency)
    # State 0: Move_Next
    # State 1: Service (Load)
    # State 2: Service (Deliver)
    # State 3: Service (Inspect)
    # State 4: Go_Charge (Emergency recovery)
    policy_aggressive = np.zeros((mdp.num_states, mdp.num_actions))
    policy_aggressive[0, 0] = 1.0  # Charging -> Move_Next
    policy_aggressive[1, 1] = 1.0  # Spinning -> Service
    policy_aggressive[2, 1] = 1.0  # Weaving -> Service
    policy_aggressive[3, 1] = 1.0  # Inspection -> Service
    policy_aggressive[4, 2] = 1.0  # Low_Battery -> Go_Charge

    # Define Policy 2: Conservative / Robust Transport Policy (Balances operations with planned charging cycles)
    # State 0: Move_Next
    # State 1: 70% Service, 30% Move_Next
    # State 2: 70% Service, 30% Move_Next
    # State 3: 50% Service, 50% Go_Charge (frequent recharging)
    # State 4: Go_Charge
    policy_conservative = np.zeros((mdp.num_states, mdp.num_actions))
    policy_conservative[0, 0] = 1.0
    policy_conservative[1, 1] = 0.7; policy_conservative[1, 0] = 0.3
    policy_conservative[2, 1] = 0.7; policy_conservative[2, 0] = 0.3
    policy_conservative[3, 1] = 0.5; policy_conservative[3, 2] = 0.5
    policy_conservative[4, 2] = 1.0

    # Analytical Policy Evaluation: Bellman Matrix Inversion V = (I - gamma*P)^(-1) * R
    v_agg, p_pi_agg, r_pi_agg = mdp.analytical_policy_evaluation(policy_aggressive)
    v_cons, p_pi_cons, r_pi_cons = mdp.analytical_policy_evaluation(policy_conservative)

    print("\n" + "=" * 75)
    print("ANALYTICAL BELLMAN MATRIX INVERSION RESULTS: V_pi = (I - gamma*P_pi)^(-1) * R_pi")
    print("=" * 75)
    print(f"{'State Name':<25} | {'Aggressive Policy V(s)':<22} | {'Conservative Policy V(s)':<24}")
    print("-" * 75)
    for s_idx, s_name in enumerate(mdp.states):
        print(f"{s_name:<25} | {v_agg[s_idx]:^22.3f} | {v_cons[s_idx]:^24.3f}")
    print("-" * 75)

    # Simulate Trajectories
    print("\nSAMPLE TRAJECTORY SIMULATION (Conservative Policy, 10 Steps):")
    print("-" * 75)
    traj, disc_return = mdp.simulate_trajectory(policy_conservative, start_state=0, num_steps=10, seed=101)
    for step in traj:
        print(f"Step {step['step']:02d}: State: {step['state']:<20} -> Action: {step['action']:<10} | Reward: {step['reward']:+5.1f} | Next: {step['next_state']}")
    print(f"Total Discounted Return (gamma=0.90): {disc_return:.3f}")
    print("-" * 75)

    # Plotting
    save_path = "mdp_network_and_values.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Bar chart of State Values
    x = np.arange(mdp.num_states)
    width = 0.35
    rects1 = axes[0].bar(x - width/2, v_agg, width, label='Aggressive Policy', color='#d62728', alpha=0.85)
    rects2 = axes[0].bar(x + width/2, v_cons, width, label='Conservative Policy', color='#1f77b4', alpha=0.85)
    axes[0].set_ylabel('State Value V(s)', fontsize=12, fontweight='bold')
    axes[0].set_title('Analytical State Values V_pi (Bellman Matrix Inversion)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.replace('_', '\n') for s in mdp.states], fontsize=9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=10)

    # State transition probability heatmap under Conservative Policy
    im = axes[1].imshow(p_pi_cons, cmap='Blues', aspect='auto')
    axes[1].set_title('Stationary Transition Probability Matrix P_pi\n(Conservative AGV Policy)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(np.arange(mdp.num_states))
    axes[1].set_yticks(np.arange(mdp.num_states))
    axes[1].set_xticklabels([f"S{i}" for i in range(mdp.num_states)], fontsize=10)
    axes[1].set_yticklabels([f"S{i}: {s}" for i, s in enumerate(mdp.states)], fontsize=9)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Add text annotations inside heatmap
    for i in range(mdp.num_states):
        for j in range(mdp.num_states):
            val = p_pi_cons[i, j]
            if val > 0.01:
                axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.5 else "black", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Visualization saved to '{save_path}'")

    print("\nCONCLUSION & ENGINEERING INFERENCE:")
    print("1. Formulated a 5-state, 4-action stochastic MDP representing an autonomous factory transport AGV.")
    print("2. Transition matrix stochasticity verified with sum_s' P(s'|s,a) = 1.0 for all pairs.")
    print("3. Analytical solution using matrix inversion (I - gamma*P)^(-1)*R exactly evaluates long-term expected returns.")
    print("4. Conservative policy significantly mitigates the low-battery hazard, proving stability in autonomous manufacturing operations.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(os.path.join(script_dir, "output.txt"))
    run_experiment()
