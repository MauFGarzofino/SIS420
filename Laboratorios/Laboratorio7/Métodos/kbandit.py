import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class Bandit:
    def __init__(self, k):
        self.k = k
        self.probabilities = np.random.rand(k)
    
    def step(self, action):
        if np.random.rand() < self.probabilities[action]:
            return 1
        else:
            return 0

def policy_gradient_bandit(k, alpha, episodes):
    bandit = Bandit(k)
    H = np.zeros(k)
    rewards = np.zeros(episodes)
    avg_rewards = []
    
    for episode in range(episodes):
        pi = softmax(H)
        action = np.random.choice(range(k), p=pi)
        reward = bandit.step(action)
        rewards[episode] = reward
        avg_reward = np.mean(rewards[:episode+1])
        avg_rewards.append(avg_reward)
        
        for i in range(k):
            if i == action:
                H[i] += alpha * (reward - avg_reward) * (1 - pi[i])
            else:
                H[i] -= alpha * (reward - avg_reward) * pi[i]

    return rewards, avg_rewards

k = 10
alpha = 0.1
episodes = 10000
rewards, avg_rewards = policy_gradient_bandit(k, alpha, episodes)

plt.plot(avg_rewards)
plt.xlabel('Episodios')
plt.ylabel('Recompensa Promedio')
plt.title('Recompensa Promedio por Episodio en K-Armed Bandit')
plt.show()
