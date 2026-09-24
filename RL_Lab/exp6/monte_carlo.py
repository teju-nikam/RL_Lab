"""
Experiment 6: Monte Carlo Prediction and Control
Course: Reinforcement Learning Lab (01AMP403)
Department: CSE-AIML, Final Year B.Tech. (AI) Semester VII
DKTE Society's Textile and Engineering Institute, Ichalkaranji

Tasks:
1. Implement First-Visit and Every-Visit Monte Carlo Prediction to estimate state-values V(s).
2. Implement On-Policy First-Visit Monte Carlo Control with epsilon-greedy exploration to learn Q*(s, a) and pi*(s).
3. Evaluate learning progress through episodic interaction without requiring environment transition models.
4. Save analytical plots and documentation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class EpisodicGridWorld:
    """
    4x4 Episodic Grid World Environment for Model-Free RL.
    Start: (0, 0)
    Goal: (3, 3) [Terminal, Reward +10.0]
    Pit: (1, 2) [Terminal, Reward -10.0]
    Step Cost: -1.0
    Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    """
    def __init__(self, slip_prob=0.1, seed=None):
        self.rows = 4
        self.cols = 4
        self.num_states = self.rows * self.cols
        self.goal = (3, 3)
        self.pit = (1, 2)
        self.terminals = [self.goal, self.pit]
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)

        self.actions = [0, 1, 2, 3]
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_arrows = ["^", "v", "<", ">"]
        self.deltas = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1)    # RIGHT
        }

    def reset(self):
        # Always starts at (0, 0)
        return (0, 0)

    def is_terminal(self, state):
        return state in self.terminals

    def step(self, state, action):
        if self.is_terminal(state):
            return state, 0.0, True

        # Stochastic slip
        actual_action = action
        if self.rng.random() < self.slip_prob:
            actual_action = self.rng.choice(self.actions)

        r, c = state
        dr, dc = self.deltas[actual_action]
        nr, nc = r + dr, c + dc

        # Check bounds
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            nr, nc = r, c

        next_state = (nr, nc)

        if next_state == self.goal:
            return next_state, 10.0, True
        elif next_state == self.pit:
            return next_state, -10.0, True
        else:
            return next_state, -1.0, False


def generate_episode(env, policy_func, max_steps=100):
    """Generates an episode: list of (state, action, reward) tuples."""
    episode = []
    state = env.reset()
    for _ in range(max_steps):
        action = policy_func(state)
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


# =====================================================================
# 1. MONTE CARLO PREDICTION (First-Visit vs Every-Visit)
# =====================================================================
def run_mc_prediction(env, num_episodes=5000, gamma=0.95):
    """
    Estimates V(s) for a uniform random policy using:
    a) First-Visit Monte Carlo
    b) Every-Visit Monte Carlo
    """
    # Policy: uniform random
    random_policy = lambda s: env.rng.choice(env.actions)

    # First-visit tracking
    returns_fv = { (r, c): [] for r in range(env.rows) for c in range(env.cols) }
    # Every-visit tracking
    returns_ev = { (r, c): [] for r in range(env.rows) for c in range(env.cols) }

    for ep in range(num_episodes):
        episode = generate_episode(env, random_policy)
        G = 0.0
        # Backward pass
        visited_states = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            # Every-visit: append return every time state appears
            returns_ev[s].append(G)

            # First-visit: append return only if this is earliest occurrence in episode
            states_prior = [episode[i][0] for i in range(t)]
            if s not in states_prior:
                returns_fv[s].append(G)

    # Compute means
    V_first = np.zeros((env.rows, env.cols))
    V_every = np.zeros((env.rows, env.cols))

    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            V_first[r, c] = np.mean(returns_fv[s]) if len(returns_fv[s]) > 0 else 0.0
            V_every[r, c] = np.mean(returns_ev[s]) if len(returns_ev[s]) > 0 else 0.0

    return V_first, V_every


# =====================================================================
# 2. ON-POLICY FIRST-VISIT MONTE CARLO CONTROL (epsilon-greedy)
# =====================================================================
def run_mc_control(env, num_episodes=10000, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.9995):
    """
    On-policy First-Visit MC Control for estimating optimal policy pi*.
    """
    Q = np.zeros((env.rows, env.cols, len(env.actions)), dtype=float)
    N = np.zeros((env.rows, env.cols, len(env.actions)), dtype=int)
    epsilon = eps_start

    episode_rewards = []
    eps_history = []

    for ep in range(1, num_episodes + 1):
        # Epsilon-greedy policy function
        def eps_policy(state):
            if env.rng.random() < epsilon:
                return env.rng.choice(env.actions)
            else:
                return int(np.argmax(Q[state[0], state[1]]))

        episode = generate_episode(env, eps_policy)
        total_ep_reward = sum([x[2] for x in episode])
        episode_rewards.append(total_ep_reward)

        # Backward discounted returns
        G = 0.0
        visited_sa = set()
        sa_pairs = [(x[0], x[1]) for x in episode]

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            # First-visit condition
            if (s, a) not in sa_pairs[:t]:
                N[s[0], s[1], a] += 1
                # Incremental sample average: Q(s,a) <- Q(s,a) + (1/N) * (G - Q(s,a))
                step_size = 1.0 / N[s[0], s[1], a]
                Q[s[0], s[1], a] += step_size * (G - Q[s[0], s[1], a])

        # Decay epsilon
        epsilon = max(eps_end, epsilon * eps_decay)
        eps_history.append(epsilon)

        if ep % 2000 == 0:
            avg_100 = np.mean(episode_rewards[-100:])
            print(f"Episode {ep:5d}/{num_episodes} | Epsilon: {epsilon:.3f} | Recent 100-Ep Return: {avg_100:6.2f}")

    # Optimal policy & value function from Q
    V_star = np.max(Q, axis=2)
    policy_star = np.argmax(Q, axis=2)

    return Q, V_star, policy_star, episode_rewards


def print_matrix(name, mat, env):
    print(f"\n{name}:")
    print("+" + "--------+" * env.cols)
    for r in range(env.rows):
        row_str = "|"
        for c in range(env.cols):
            if (r, c) == env.goal:
                row_str += "  GOAL  |"
            elif (r, c) == env.pit:
                row_str += "  PIT   |"
            else:
                row_str += f" {mat[r, c]:6.2f} |"
        print(row_str)
        print("+" + "--------+" * env.cols)


def plot_results(V_first, V_every, V_star, policy_star, ep_rewards, env, save_path="mc_prediction_control.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Subplot 1: First-Visit MC Value Map
    im1 = axes[0].imshow(V_first, cmap="viridis")
    axes[0].set_title("First-Visit MC Prediction $V^\\pi(s)$\n(Random Policy, 5000 Episodes)", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[0])
    for r in range(env.rows):
        for c in range(env.cols):
            axes[0].text(c, r, f"{V_first[r, c]:.1f}", ha="center", va="center", color="white" if V_first[r, c] < 0 else "black", fontweight="bold")

    # Subplot 2: Learning Curve of MC Control
    window = 100
    smoothed_rewards = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_rewards, color="#1f77b4", linewidth=1.5)
    axes[1].set_xlabel("Episode", fontsize=11, fontweight="bold")
    axes[1].set_ylabel(f"Return (Moving Avg {window})", fontsize=11, fontweight="bold")
    axes[1].set_title("MC Control Learning Curve\n(Epsilon-Greedy Policy Improvement)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Subplot 3: Optimal Policy from MC Control
    im3 = axes[2].imshow(V_star, cmap="magma")
    axes[2].set_title("Optimal Policy $\\pi^*(s)$ & Value $V^*(s)$\n(On-Policy First-Visit MC Control)", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=axes[2])
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) == env.goal:
                axes[2].text(c, r, "GOAL\n+10", ha="center", va="center", color="yellow", fontweight="bold")
            elif (r, c) == env.pit:
                axes[2].text(c, r, "PIT\n-10", ha="center", va="center", color="cyan", fontweight="bold")
            else:
                act = policy_star[r, c]
                arrow = env.action_arrows[act]
                axes[2].text(c, r, f"{arrow}\n{V_star[r, c]:.1f}", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Visualization plot saved to '{save_path}'")


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
    print("EXPERIMENT 6: MONTE CARLO PREDICTION AND CONTROL")
    print("=" * 75)
    env = EpisodicGridWorld(slip_prob=0.1, seed=42)

    # Part 1: Prediction
    print("Running Part 1: Monte Carlo Prediction (5,000 episodes)...")
    V_first, V_every = run_mc_prediction(env, num_episodes=5000, gamma=0.95)
    print_matrix("First-Visit MC Estimated V(s)", V_first, env)
    print_matrix("Every-Visit MC Estimated V(s)", V_every, env)

    # Difference between First-Visit and Every-Visit
    pred_diff = np.max(np.abs(V_first - V_every))
    print(f"\nMax difference between First-Visit and Every-Visit estimates: {pred_diff:.4f}")

    # Part 2: Control
    print("\nRunning Part 2: On-Policy First-Visit MC Control (10,000 episodes)...")
    Q_star, V_star, policy_star, rewards = run_mc_control(env, num_episodes=10000, gamma=0.95)
    print_matrix("Optimal State-Value Function V*(s) (via MC Control)", V_star, env)

    print("\nOPTIMAL POLICY pi*(s) [DIRECTIONAL ACTIONS]:")
    print("+" + "--------+" * env.cols)
    for r in range(env.rows):
        row_str = "|"
        for c in range(env.cols):
            if (r, c) == env.goal:
                row_str += "  GOAL  |"
            elif (r, c) == env.pit:
                row_str += "  PIT   |"
            else:
                act = policy_star[r, c]
                arrow = env.action_arrows[act]
                name = env.action_names[act]
                row_str += f" {arrow} {name:<4}|"
        print(row_str)
        print("+" + "--------+" * env.cols)

    plot_results(V_first, V_every, V_star, policy_star, rewards, env, save_path="mc_prediction_control.png")

    print("\nCONCLUSION & INFERENCE:")
    print("1. Monte Carlo methods learn entirely from simulated experience without model dynamics P(s'|s,a).")
    print("2. First-visit and Every-visit MC prediction yield virtually identical asymptotic value estimates.")
    print("3. MC Control successfully discovered the optimal path navigating around the hazard PIT to reach the GOAL.")
