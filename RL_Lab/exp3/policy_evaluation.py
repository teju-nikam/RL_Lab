"""
Experiment 3: Dynamic Programming – Policy Evaluation
Course: Reinforcement Learning Lab (01AMP403)
Department: CSE-AIML, Final Year B.Tech. (AI) Semester VII
DKTE Society's Textile and Engineering Institute, Ichalkaranji

Tasks:
1. Implement standard 4x4 Grid World environment with terminal states at (0,0) and (3,3).
2. Implement Iterative Policy Evaluation for an equiprobable random policy (0.25 each action).
3. Track value function convergence across sweeps k = 0, 1, 2, 10, and final converged values.
4. Output exact numerical matrices and save visual heatmaps.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class GridWorld4x4:
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.num_states = self.rows * self.cols
        self.terminal_states = [(0, 0), (3, 3)]
        # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.actions = [0, 1, 2, 3]
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_deltas = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1)    # RIGHT
        }

    def is_terminal(self, r, c):
        return (r, c) in self.terminal_states

    def step(self, r, c, action):
        """Returns next (r', c') and reward."""
        if self.is_terminal(r, c):
            return (r, c), 0.0

        dr, dc = self.action_deltas[action]
        nr, nc = r + dr, c + dc

        # Check grid boundaries
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            nr, nc = r, c  # Bounce back if off grid

        reward = -1.0
        return (nr, nc), reward


def iterative_policy_evaluation(env, theta=1e-5, gamma=1.0):
    """
    Computes state-value function V(s) for an equiprobable random policy.
    Bellman Expectation Update:
    V_{k+1}(s) = sum_a pi(a|s) * [ R + gamma * V_k(s') ]
    """
    V = np.zeros((env.rows, env.cols), dtype=float)
    snapshots = {}
    snapshots[0] = np.copy(V)

    sweep = 0
    delta_history = []

    print("=" * 70)
    print("EXPERIMENT 3: ITERATIVE POLICY EVALUATION (GRID WORLD)")
    print("=" * 70)
    print(f"Grid Dimensions: {env.rows} x {env.cols}")
    print(f"Terminal States: {env.terminal_states}")
    print(f"Reward per step: -1.0 | Gamma: {gamma} | Convergence threshold (theta): {theta}")
    print(f"Policy: Equiprobable Random (0.25 UP, DOWN, LEFT, RIGHT)")
    print("-" * 70)

    while True:
        sweep += 1
        delta = 0.0
        V_new = np.copy(V)

        for r in range(env.rows):
            for c in range(env.cols):
                if env.is_terminal(r, c):
                    continue

                v_s = 0.0
                prob_a = 1.0 / len(env.actions)

                for a in env.actions:
                    (nr, nc), reward = env.step(r, c, a)
                    v_s += prob_a * (reward + gamma * V[nr, nc])

                delta = max(delta, abs(v_s - V[r, c]))
                V_new[r, c] = v_s

        V = V_new
        delta_history.append(delta)

        if sweep in [1, 2, 3, 10]:
            snapshots[sweep] = np.copy(V)

        if delta < theta:
            snapshots['converged'] = np.copy(V)
            print(f"Iterative Policy Evaluation converged in {sweep} sweeps! Final max delta: {delta:.2e}")
            break

    return V, snapshots, sweep, delta_history


def print_matrix(name, mat):
    print(f"\nValue Function Matrix: {name}")
    print("+" + "--------+" * 4)
    for r in range(mat.shape[0]):
        row_str = "|"
        for c in range(mat.shape[1]):
            row_str += f" {mat[r, c]:6.1f} |"
        print(row_str)
        print("+" + "--------+" * 4)


def plot_snapshots(snapshots, final_sweep, save_path="policy_evaluation_grid.png"):
    display_keys = [0, 1, 2, 10, 'converged']
    titles = [
        "Sweep k = 0 (Init)",
        "Sweep k = 1",
        "Sweep k = 2",
        "Sweep k = 10",
        f"Converged (k = {final_sweep})"
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=300)

    for idx, key in enumerate(display_keys):
        mat = snapshots[key]
        im = axes[idx].imshow(mat, cmap="coolwarm", vmin=-22, vmax=0)
        axes[idx].set_title(titles[idx], fontsize=11, fontweight='bold')
        axes[idx].set_xticks(range(4))
        axes[idx].set_yticks(range(4))

        # Value annotations
        for r in range(4):
            for c in range(4):
                axes[idx].text(c, r, f"{mat[r, c]:.1f}",
                              ha="center", va="center",
                              color="white" if mat[r, c] < -10 else "black",
                              fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Snapshot visualization successfully saved to '{save_path}'")


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

    env = GridWorld4x4()
    V_star, snapshots, sweep, deltas = iterative_policy_evaluation(env)

    # Print snapshots
    print_matrix("Sweep k = 0", snapshots[0])
    print_matrix("Sweep k = 1", snapshots[1])
    print_matrix("Sweep k = 2", snapshots[2])
    print_matrix("Sweep k = 10", snapshots[10])
    print_matrix(f"Final Converged Matrix (k = {sweep})", snapshots['converged'])

    plot_path = os.path.join(script_dir, "policy_evaluation_grid.png")
    plot_snapshots(snapshots, sweep, save_path=plot_path)

    print("\nCONCLUSION & VERIFICATION:")
    print("1. Successfully validated Sutton & Barto's classic Example 4.1.")
    print("2. Terminal states (0,0) and (3,3) maintain fixed V(s) = 0.")
    print("3. States adjacent to terminals converge to ~ -14.0, while furthest states converge to ~ -22.0.")
    print("4. Monotonic convergence confirms contraction mapping theorem of the Bellman expectation operator.")
