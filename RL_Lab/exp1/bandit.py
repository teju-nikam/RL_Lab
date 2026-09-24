"""
Experiment 1: Multi-Armed Bandit Problem
Course: Reinforcement Learning Lab (01AMP403)
Department: CSE-AIML, Final Year B.Tech. (AI) Semester VII
DKTE Society's Textile and Engineering Institute, Ichalkaranji

Tasks:
1. Implement the 10-armed bandit testbed.
2. Implement epsilon-greedy action selection (greedy, eps=0.01, eps=0.1, decaying eps).
3. Implement Upper Confidence Bound (UCB) action selection.
4. Compare exploration and exploitation strategies over repeated runs.
5. Save visualization plots and summary metrics to output files.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit:
    """
    10-armed Testbed environment following Sutton & Barto Chapter 2.
    True action values q*(a) are sampled from N(0, 1).
    Actual reward when action a is selected is sampled from N(q*(a), 1).
    """
    def __init__(self, k=10, seed=None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        # True value of each action
        self.q_star = self.rng.normal(0.0, 1.0, size=self.k)
        self.optimal_action = np.argmax(self.q_star)

    def step(self, action):
        # Reward is drawn from normal distribution with mean q*(action) and variance 1
        reward = self.rng.normal(self.q_star[action], 1.0)
        is_optimal = (action == self.optimal_action)
        return reward, is_optimal


class EpsilonGreedyAgent:
    """
    Epsilon-greedy agent with sample-average incremental action-value estimation.
    """
    def __init__(self, k=10, epsilon=0.1, decay=False, decay_rate=0.005, name=None):
        self.k = k
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.decay = decay
        self.decay_rate = decay_rate
        self.name = name or f"eps={epsilon}"
        self.reset()

    def reset(self):
        self.q_estimates = np.zeros(self.k, dtype=float)
        self.action_counts = np.zeros(self.k, dtype=int)
        self.t = 0
        self.epsilon = self.epsilon_init

    def select_action(self, rng):
        self.t += 1
        if self.decay:
            # Epsilon decay: epsilon_t = epsilon_0 / (1 + decay_rate * t)
            self.epsilon = self.epsilon_init / (1.0 + self.decay_rate * self.t)

        if rng.random() < self.epsilon:
            # Exploration: choose random action uniformly
            return rng.integers(0, self.k)
        else:
            # Exploitation: choose greedy action with random tie-breaking
            max_val = np.max(self.q_estimates)
            best_actions = np.where(self.q_estimates == max_val)[0]
            return rng.choice(best_actions)

    def update(self, action, reward):
        self.action_counts[action] += 1
        # Incremental sample average update: Q(a) <- Q(a) + (1/N(a)) * [R - Q(a)]
        step_size = 1.0 / self.action_counts[action]
        self.q_estimates[action] += step_size * (reward - self.q_estimates[action])


class UCBAgent:
    """
    Upper Confidence Bound (UCB1) agent.
    A_t = argmax_a [ Q_t(a) + c * sqrt(ln(t) / N_t(a)) ]
    """
    def __init__(self, k=10, c=2.0, name=None):
        self.k = k
        self.c = c
        self.name = name or f"UCB (c={c})"
        self.reset()

    def reset(self):
        self.q_estimates = np.zeros(self.k, dtype=float)
        self.action_counts = np.zeros(self.k, dtype=int)
        self.t = 0

    def select_action(self, rng):
        self.t += 1
        # Try every unselected action at least once
        for a in range(self.k):
            if self.action_counts[a] == 0:
                return a
        # UCB calculation
        bonus = self.c * np.sqrt(np.log(self.t) / self.action_counts)
        ucb_values = self.q_estimates + bonus
        max_val = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_val)[0]
        return rng.choice(best_actions)

    def update(self, action, reward):
        self.action_counts[action] += 1
        step_size = 1.0 / self.action_counts[action]
        self.q_estimates[action] += step_size * (reward - self.q_estimates[action])


def run_experiment(num_runs=1000, num_steps=2000, seed=42):
    """
    Simulates multiple independent runs of the 10-armed bandit testbed
    across multiple exploration-exploitation strategies.
    """
    master_rng = np.random.default_rng(seed)

    agents = [
        EpsilonGreedyAgent(k=10, epsilon=0.0, name="Greedy (eps=0.0)"),
        EpsilonGreedyAgent(k=10, epsilon=0.01, name="eps-Greedy (eps=0.01)"),
        EpsilonGreedyAgent(k=10, epsilon=0.1, name="eps-Greedy (eps=0.10)"),
        EpsilonGreedyAgent(k=10, epsilon=0.2, decay=True, decay_rate=0.005, name="Decaying eps (eps0=0.20)"),
        UCBAgent(k=10, c=2.0, name="UCB (c=2.0)")
    ]

    # Arrays to accumulate metrics: (num_agents, num_runs, num_steps)
    rewards = np.zeros((len(agents), num_runs, num_steps), dtype=float)
    optimal_actions = np.zeros((len(agents), num_runs, num_steps), dtype=float)

    print("=" * 70)
    print("EXPERIMENT 1: MULTI-ARMED BANDIT EXPLORATION VS EXPLOITATION")
    print("=" * 70)
    print(f"Number of Arms (k): 10")
    print(f"Number of Independent Runs: {num_runs}")
    print(f"Steps per Run: {num_steps}")
    print("Strategies tested:")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent.name}")
    print("-" * 70)

    for r in range(num_runs):
        if (r + 1) % 200 == 0 or r == 0:
            print(f"Simulating run {r+1}/{num_runs}...")
        run_seed = master_rng.integers(0, 10**8)
        # Create a bandit instance for this run
        bandit = MultiArmedBandit(k=10, seed=run_seed)

        for a_idx, agent in enumerate(agents):
            agent.reset()
            agent_rng = np.random.default_rng(run_seed + a_idx + 1)
            for step in range(num_steps):
                action = agent.select_action(agent_rng)
                reward, is_optimal = bandit.step(action)
                agent.update(action, reward)
                rewards[a_idx, r, step] = reward
                optimal_actions[a_idx, r, step] = 1.0 if is_optimal else 0.0

    print("Simulation completed successfully!")
    print("-" * 70)

    # Compute averages across runs
    avg_rewards = np.mean(rewards, axis=1)  # shape: (num_agents, num_steps)
    pct_optimal = np.mean(optimal_actions, axis=1) * 100.0  # percentage

    # Print summary performance metrics
    print("\nPERFORMANCE SUMMARY TABLE (Averaged over 1000 runs):")
    print(f"{'Strategy':<26} | {'Avg Reward (Final 100)':<22} | {'Optimal Action % (Final)':<22}")
    print("-" * 76)
    summary_data = []
    for a_idx, agent in enumerate(agents):
        final_rew = np.mean(avg_rewards[a_idx, -100:])
        final_opt = np.mean(pct_optimal[a_idx, -100:])
        print(f"{agent.name:<26} | {final_rew:^22.4f} | {final_opt:^22.2f}%")
        summary_data.append((agent.name, final_rew, final_opt))
    print("-" * 76)

    return agents, avg_rewards, pct_optimal, summary_data


def plot_results(agents, avg_rewards, pct_optimal, save_path="bandit_comparison.png"):
    """
    Plots the average reward and % optimal action over time.
    """
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    styles = ['-', '--', '-', '-.', ':']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300)

    # Subplot 1: Average Reward
    for i, agent in enumerate(agents):
        axes[0].plot(avg_rewards[i], label=agent.name, color=colors[i], linestyle=styles[i], linewidth=1.8)
    axes[0].set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    axes[0].set_title('10-Armed Bandit: Exploration vs Exploitation (Average Reward)', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right', fontsize=10)

    # Subplot 2: % Optimal Action
    for i, agent in enumerate(agents):
        axes[1].plot(pct_optimal[i], label=agent.name, color=colors[i], linestyle=styles[i], linewidth=1.8)
    axes[1].set_xlabel('Steps', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('% Optimal Action', fontsize=12, fontweight='bold')
    axes[1].set_title('10-Armed Bandit: % Optimal Action Selection', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Visualization plot successfully saved to '{save_path}'")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Output redirection to capture both console and file
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

    agents, avg_rewards, pct_optimal, summary = run_experiment(num_runs=1000, num_steps=2000, seed=42)
    plot_path = os.path.join(script_dir, "bandit_comparison.png")
    plot_results(agents, avg_rewards, pct_optimal, save_path=plot_path)

    print("\nCONCLUSION & INFERENCE:")
    print("1. Pure Greedy (eps=0.0) gets stuck in sub-optimal actions very early, locking into ~35-40% optimal actions.")
    print("2. eps-Greedy (eps=0.10) explores rapidly and finds the optimal arm faster, achieving high rewards quickly.")
    print("3. eps-Greedy (eps=0.01) explores slowly, eventually outperforming eps=0.10 in the long run because it exploits 99% of the time.")
    print("4. Decaying eps-Greedy bridges both regimes: fast initial exploration followed by near-complete exploitation.")
    print("5. UCB (c=2.0) achieves top performance by using uncertainty-aware exploration bonuses based on visitation counts.")
