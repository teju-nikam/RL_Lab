"""
Experiment 7: Temporal Difference (TD) Learning
Course: Reinforcement Learning Lab (01AMP403)
Department: CSE-AIML, Final Year B.Tech. (AI) Semester VII
DKTE Society's Textile and Engineering Institute, Ichalkaranji

Tasks:
1. Implement the classic 5-State Random Walk benchmark (Sutton & Barto Chapter 6).
2. Implement TD(0) prediction using the one-step bootstrapping update.
3. Implement Constant-alpha Monte Carlo prediction.
4. Compare TD(0) vs. Monte Carlo across multiple learning rates alpha:
   - Value function convergence profiles
   - Root Mean Square Error (RMSE) against analytical true state values over 100 runs.
5. Save comparison plots and documentation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class RandomWalkEnvironment:
    """
    5-State Random Walk Markov Reward Process (Sutton & Barto Example 6.2).
    States: 0: A, 1: B, 2: C (start), 3: D, 4: E
    Left terminal (-1): reward 0
    Right terminal (5): reward +1
    All non-terminal transitions yield reward 0.
    """
    def __init__(self, seed=None):
        self.states = ["A", "B", "C", "D", "E"]
        self.num_states = len(self.states)
        self.start_state = 2  # State C
        self.rng = np.random.default_rng(seed)
        # Analytical true values under equiprobable random walk (gamma=1.0)
        self.true_values = np.array([1.0/6.0, 2.0/6.0, 3.0/6.0, 4.0/6.0, 5.0/6.0])

    def step(self, state):
        """Random step left (-1) or right (+1) with equal probability 0.5."""
        action = self.rng.choice([-1, 1])
        next_state = state + action
        if next_state == -1:
            return next_state, 0.0, True
        elif next_state == self.num_states:
            return next_state, 1.0, True
        else:
            return next_state, 0.0, False


def run_td0_single_run(env, num_episodes=100, alpha=0.1, gamma=1.0):
    """Executes TD(0) prediction for one run and records intermediate V after episodes."""
    V = np.full(env.num_states, 0.5, dtype=float)
    snapshots = { 0: np.copy(V) }
    tracked_episodes = [1, 10, 25, 100]

    for ep in range(1, num_episodes + 1):
        state = env.start_state
        while True:
            next_state, reward, done = env.step(state)
            next_val = 0.0 if done else V[next_state]
            # TD Error: delta = R + gamma * V(S') - V(S)
            td_target = reward + gamma * next_val
            V[state] += alpha * (td_target - V[state])
            if done:
                break
            state = next_state

        if ep in tracked_episodes:
            snapshots[ep] = np.copy(V)

    return V, snapshots


def compute_rmse_comparison(num_runs=100, num_episodes=100, seed=42):
    """
    Computes empirical RMSE across episodes for TD(0) and Constant-alpha MC
    under varying alpha parameters, averaged over 100 independent runs.
    """
    master_rng = np.random.default_rng(seed)
    td_alphas = [0.05, 0.10, 0.15]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]

    td_errors = { a: np.zeros(num_episodes) for a in td_alphas }
    mc_errors = { a: np.zeros(num_episodes) for a in mc_alphas }

    env = RandomWalkEnvironment()
    true_v = env.true_values

    print("Computing RMSE comparison over 100 independent runs...")
    for r in range(num_runs):
        run_seed = master_rng.integers(0, 10**8)
        run_env = RandomWalkEnvironment(seed=run_seed)

        # Pre-generate episodes for this run to compare fairly
        episodes = []
        for _ in range(num_episodes):
            ep = []
            s = run_env.start_state
            while True:
                ns, rew, done = run_env.step(s)
                ep.append((s, rew, ns, done))
                if done:
                    break
                s = ns
            episodes.append(ep)

        # 1. Evaluate TD(0)
        for alpha in td_alphas:
            V = np.full(run_env.num_states, 0.5, dtype=float)
            for ep_idx, ep in enumerate(episodes):
                for s, rew, ns, done in ep:
                    next_val = 0.0 if done else V[ns]
                    V[s] += alpha * (rew + next_val - V[s])
                rmse = np.sqrt(np.mean((V - true_v) ** 2))
                td_errors[alpha][ep_idx] += rmse

        # 2. Evaluate Constant-alpha Monte Carlo
        for alpha in mc_alphas:
            V = np.full(run_env.num_states, 0.5, dtype=float)
            for ep_idx, ep in enumerate(episodes):
                # In random walk, the entire episode return G is simply the final reward
                final_return = ep[-1][1]
                # Constant-alpha MC updates each visited state towards G
                visited = set()
                for s, _, _, _ in ep:
                    if s not in visited:
                        visited.add(s)
                        V[s] += alpha * (final_return - V[s])
                rmse = np.sqrt(np.mean((V - true_v) ** 2))
                mc_errors[alpha][ep_idx] += rmse

    # Average over runs
    for a in td_alphas:
        td_errors[a] /= num_runs
    for a in mc_alphas:
        mc_errors[a] /= num_runs

    return td_errors, mc_errors


def plot_td_results(env, snapshots, td_errors, mc_errors, save_path="td_vs_mc_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Subplot 1: Value Estimates after varying episodes (Sutton & Barto Fig 6.2 Left)
    state_names = env.states
    x = np.arange(len(state_names))

    axes[0].plot(x, env.true_values, 'k-', linewidth=2.5, marker='o', label='True Values')
    for ep, vals in sorted(snapshots.items()):
        axes[0].plot(x, vals, linestyle='--', marker='s', label=f'{ep} Episodes' if ep > 0 else 'Init (0 Ep)')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(state_names, fontsize=11, fontweight='bold')
    axes[0].set_xlabel('State', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Estimated Value V(s)', fontsize=12, fontweight='bold')
    axes[0].set_title('TD(0) Value Function Progression ($\\alpha = 0.1$)', fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Subplot 2: Empirical RMSE Comparison (TD vs MC)
    for alpha, err in td_errors.items():
        axes[1].plot(err, label=f'TD(0) $\\alpha={alpha}$', linewidth=1.8)
    for alpha, err in mc_errors.items():
        axes[1].plot(err, linestyle=':', linewidth=1.8, label=f'MC $\\alpha={alpha}$')

    axes[1].set_xlabel('Walks / Episodes', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Empirical RMS Error', fontsize=12, fontweight='bold')
    axes[1].set_title('Empirical RMS Error: TD(0) vs. Monte Carlo (100 Runs)', fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Visualization plot successfully saved to '{save_path}'")


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

    print("=" * 75)
    print("EXPERIMENT 7: TEMPORAL DIFFERENCE (TD) LEARNING VS. MONTE CARLO")
    print("ENVIRONMENT: 5-STATE RANDOM WALK MRP")
    print("=" * 75)

    env = RandomWalkEnvironment(seed=42)
    print(f"States: {env.states} (A:0, B:1, C:2, D:3, E:4)")
    print("True Analytical Values:")
    for s_name, val in zip(env.states, env.true_values):
        print(f"  V({s_name}) = {val:.4f}")
    print("-" * 75)

    # 1. Run single instance of TD(0)
    V_final, snapshots = run_td0_single_run(env, num_episodes=100, alpha=0.1)

    print("\nTD(0) VALUE ESTIMATES ACROSS EPISODES (alpha = 0.1):")
    print(f"{'State':<8} | {'True Val':<10} | {'0 Ep':<8} | {'1 Ep':<8} | {'10 Ep':<8} | {'25 Ep':<8} | {'100 Ep':<8}")
    print("-" * 70)
    for idx, s_name in enumerate(env.states):
        print(f"{s_name:<8} | {env.true_values[idx]:<10.4f} | {snapshots[0][idx]:<8.4f} | {snapshots[1][idx]:<8.4f} | {snapshots[10][idx]:<8.4f} | {snapshots[25][idx]:<8.4f} | {snapshots[100][idx]:<8.4f}")
    print("-" * 70)

    # 2. Compute RMSE comparison over 100 runs
    td_errs, mc_errs = compute_rmse_comparison(num_runs=100, num_episodes=100, seed=42)

    print("\nFINAL RMS ERROR COMPARISON (After 100 Episodes, 100 Runs):")
    print(f"{'Algorithm':<25} | {'Learning Rate (alpha)':<25} | {'Final RMSE':<15}")
    print("-" * 70)
    for a, err in td_errs.items():
        print(f"{'TD(0)':<25} | {a:<25.2f} | {err[-1]:<15.4f}")
    for a, err in mc_errs.items():
        print(f"{'Monte Carlo':<25} | {a:<25.2f} | {err[-1]:<15.4f}")
    print("-" * 70)

    plot_td_results(env, snapshots, td_errs, mc_errs, save_path=os.path.join(script_dir, "td_vs_mc_comparison.png"))

    print("\nCONCLUSION & INFERENCE:")
    print("1. TD(0) updates state values online step-by-step using bootstrapping without waiting for episode termination.")
    print("2. TD(0) with alpha=0.10 and alpha=0.15 converges substantially faster and achieves lower RMS error than Constant-alpha MC.")
    print("3. Monte Carlo suffers from higher variance because updates rely on complete sample episode returns.")
    print("4. Successfully reproduced Sutton & Barto Chapter 6 empirical findings.")
