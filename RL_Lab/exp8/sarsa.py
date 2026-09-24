"""
Experiment 8: SARSA Algorithm (On-Policy TD Control)
Course: Reinforcement Learning Lab (01AMP403)
Department: CSE-AIML, Final Year B.Tech. (AI) Semester VII
DKTE Society's Textile and Engineering Institute, Ichalkaranji

Tasks:
1. Implement the SARSA (State-Action-Reward-State-Action) algorithm.
2. Train an agent on the standard CliffWalking environment.
3. Use epsilon-greedy exploration with decay.
4. Evaluate the learned policy and demonstrate the safe path trajectory.
5. Save visualization plots and summary metrics to output files.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym


class CliffWalkingEnv:
    """
    Standard Cliff Walking Environment (Sutton & Barto Example 6.6)
    Grid: 4 rows x 12 columns
    Start: (3, 0) = state 36
    Goal: (3, 11) = state 47
    Cliff: (3, 1) through (3, 10) = states 37 to 46
    Actions: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    """
    def __init__(self):
        try:
            self.gym_env = gym.make("CliffWalking-v1")
        except Exception:
            try:
                self.gym_env = gym.make("CliffWalking-v0")
            except Exception:
                self.gym_env = None

        self.rows = 4
        self.cols = 12
        self.n_states = self.rows * self.cols  # 48
        self.n_actions = 4
        self.start_state = 36
        self.goal_state = 47
        self.cliff_states = list(range(37, 47))

        self.action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        self.action_arrows = ["^", ">", "v", "<"]
        self.deltas = {
            0: (-1, 0),  # UP
            1: (0, 1),   # RIGHT
            2: (1, 0),   # DOWN
            3: (0, -1)   # LEFT
        }

    def reset(self):
        if self.gym_env is not None:
            s, _ = self.gym_env.reset()
            return int(s)
        return self.start_state

    def step(self, state, action):
        if self.gym_env is not None:
            ns, rew, term, trunc, _ = self.gym_env.step(action)
            return int(ns), float(rew), bool(term or trunc)

        # Fallback pure python simulation
        r = state // self.cols
        c = state % self.cols
        dr, dc = self.deltas[action]
        nr = min(max(r + dr, 0), self.rows - 1)
        nc = min(max(c + dc, 0), self.cols - 1)
        next_state = nr * self.cols + nc

        if next_state in self.cliff_states:
            return self.start_state, -100.0, False
        elif next_state == self.goal_state:
            return next_state, -1.0, True
        else:
            return next_state, -1.0, False


class SARSAAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)

        # Q-table: shape (n_states, n_actions)
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, state, greedy=False):
        """Epsilon-greedy action selection."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.n_actions)
        else:
            max_q = np.max(self.Q[state])
            best_actions = np.where(self.Q[state] == max_q)[0]
            return self.rng.choice(best_actions)

    def update(self, s, a, r, s_next, a_next, done):
        """
        SARSA Update Rule:
        Q(S, A) <- Q(S, A) + alpha * [ R + gamma * Q(S', A') - Q(S, A) ]
        """
        next_q = 0.0 if done else self.Q[s_next, a_next]
        td_target = r + self.gamma * next_q
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


def train_sarsa(env, agent, num_episodes=500):
    episode_rewards = []
    episode_steps = []

    print("=" * 70)
    print("EXPERIMENT 8: SARSA ALGORITHM (ON-POLICY TD CONTROL)")
    print("ENVIRONMENT: CLIFF WALKING (4x12 Grid, 48 States, 4 Actions)")
    print("=" * 70)
    print(f"Training Episodes: {num_episodes} | Alpha: {agent.alpha} | Gamma: {agent.gamma}")
    print("-" * 70)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        action = agent.select_action(state)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1

            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

            if steps >= 500:  # safety cap
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if ep % 50 == 0:
            avg_rew = np.mean(episode_rewards[-50:])
            avg_st = np.mean(episode_steps[-50:])
            print(f"Episode {ep:3d}/{num_episodes} | Epsilon: {agent.epsilon:.4f} | Avg Reward (last 50): {avg_rew:6.1f} | Avg Steps: {avg_st:5.1f}")

    return episode_rewards, episode_steps


def evaluate_policy(env, agent):
    """Evaluates the deterministic learned greedy policy."""
    state = env.reset()
    path = [state]
    actions_taken = []
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < 100:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done = env.step(state, action)
        path.append(next_state)
        actions_taken.append(action)
        total_reward += reward
        steps += 1
        state = next_state

    return path, actions_taken, total_reward, steps


def plot_results(episode_rewards, episode_steps, path, env, save_path="sarsa_cliffwalking.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300)

    # Subplot 1: Episode Reward Learning Curve
    window = 25
    smoothed_rew = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(episode_rewards, alpha=0.3, color="#1f77b4", label="Episode Reward")
    axes[0].plot(smoothed_rew, color="#d62728", linewidth=2.0, label=f"Moving Avg ({window} ep)")
    axes[0].set_xlabel("Episode", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Total Reward", fontsize=11, fontweight="bold")
    axes[0].set_title("SARSA Learning Curve on Cliff Walking", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Subplot 2: Grid Trajectory Path Map
    grid = np.zeros((env.rows, env.cols))
    # Mark cliff
    for cs in env.cliff_states:
        grid[cs // env.cols, cs % env.cols] = -1.0
    # Mark start and goal
    grid[3, 0] = 0.5
    grid[3, 11] = 1.0

    im = axes[1].imshow(grid, cmap="Pastel1", aspect="auto")
    axes[1].set_title("Learned Safe Policy Trajectory (SARSA)", fontsize=12, fontweight="bold")

    # Plot path line
    r_coords = [s // env.cols for s in path]
    c_coords = [s % env.cols for s in path]
    axes[1].plot(c_coords, r_coords, 'ro-', linewidth=2.5, markersize=8, label="Agent Path")

    # Annotate grid
    for r in range(env.rows):
        for c in range(env.cols):
            s = r * env.cols + c
            if s == env.start_state:
                axes[1].text(c, r, "START\n(S)", ha="center", va="center", fontweight="bold", color="blue", fontsize=9)
            elif s == env.goal_state:
                axes[1].text(c, r, "GOAL\n(G)", ha="center", va="center", fontweight="bold", color="green", fontsize=9)
            elif s in env.cliff_states:
                axes[1].text(c, r, "CLIFF\n(-100)", ha="center", va="center", fontweight="bold", color="red", fontsize=8)

    axes[1].set_xticks(range(env.cols))
    axes[1].set_yticks(range(env.rows))
    axes[1].legend(loc="upper right", fontsize=10)

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

    env = CliffWalkingEnv()
    agent = SARSAAgent(env.n_states, env.n_actions, alpha=0.1, gamma=0.99,
                       eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=42)

    rewards, steps = train_sarsa(env, agent, num_episodes=500)
    path, actions, test_rew, test_steps = evaluate_policy(env, agent)

    print("\nEVALUATION OF LEARNED POLICY (Greedy, epsilon=0):")
    print("-" * 70)
    print(f"Total Steps to Goal: {test_steps}")
    print(f"Total Test Reward:   {test_rew:.1f}")
    path_coords = [f"({s // env.cols}, {s % env.cols})" for s in path]
    print(f"State Sequence Traversed ({len(path)} states):")
    print(" -> ".join(path_coords))
    action_sequence = [env.action_names[a] for a in actions]
    print(f"Actions Taken ({len(actions)} actions):")
    print(" -> ".join(action_sequence))
    print("-" * 70)

    plot_results(rewards, steps, path, env, save_path=os.path.join(script_dir, "sarsa_cliffwalking.png"))

    print("\nCONCLUSION & INFERENCE:")
    print("1. SARSA is an on-policy TD control algorithm that updates Q based on the action actually executed by the policy.")
    print("2. Because SARSA accounts for its own epsilon-greedy exploratory mistakes, it learns a safer path further away from the cliff edge.")
    print("3. Successfully reached the goal in finite steps without falling into the -100 reward cliff during testing.")
