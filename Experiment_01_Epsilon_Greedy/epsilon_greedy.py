import numpy as np
import matplotlib.pyplot as plt

# Number of arms
n_arms = 5

# Number of times each arm is selected
n_rounds = 1000

# Epsilon value
epsilon = 0.1

# True reward probabilities of each arm
true_probabilities = [0.2, 0.5, 0.7, 0.4, 0.9]

# Estimated value of each arm
estimated_values = np.zeros(n_arms)

# Number of selections of each arm
arm_counts = np.zeros(n_arms)

# Store rewards
rewards = []

for i in range(n_rounds):

    # Exploration vs Exploitation
    if np.random.random() < epsilon:
        # Exploration: choose a random arm
        action = np.random.randint(n_arms)
    else:
        # Exploitation: choose arm with highest estimated value
        action = np.argmax(estimated_values)

    # Generate reward
    reward = 1 if np.random.random() < true_probabilities[action] else 0

    # Update count
    arm_counts[action] += 1

    # Update estimated value
    estimated_values[action] += (
        reward - estimated_values[action]
    ) / arm_counts[action]

    # Store reward
    rewards.append(reward)

# Display results
print("True Probabilities:")
print(true_probabilities)

print("\nEstimated Values:")
print(estimated_values)

print("\nArm Selection Counts:")
print(arm_counts)

print("\nTotal Reward:")
print(sum(rewards))

print("\nAverage Reward:")
print(np.mean(rewards))

# Plot cumulative average reward
cumulative_average = np.cumsum(rewards) / np.arange(1, n_rounds + 1)

plt.plot(cumulative_average)
plt.xlabel("Number of Rounds")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Multi-Armed Bandit")
plt.grid()
plt.show()