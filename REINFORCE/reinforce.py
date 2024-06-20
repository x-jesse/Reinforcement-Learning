import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

import numpy as np

class FeedForward(nn.Module):

    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.l1 = nn.Linear(n_obs, 64)
        self.l2 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)
    
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

env = gym.make('CartPole-v1', render_mode=None)
state, info = env.reset()

max_epochs = 200
policy = FeedForward(env.observation_space.shape[0], env.action_space.n)
policy.to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

for n_epoch in range(max_epochs):
    cum_reward = 0
    states, actions, rewards = [], [], []
    saved_log_probs = []
    state, info = env.reset()

    while True:
        state = torch.tensor(state, device=device).unsqueeze(0)
        logits = policy(state)

        action_distr = Categorical(logits=logits)
        action = action_distr.sample()
        actions.append(action)
        saved_log_probs.append(action_distr.log_prob(action))
        
        new_state, reward, terminated, truncated, info = env.step(action.item())
        states.append(new_state)
        rewards.append(reward)
        cum_reward += reward
        
        state = new_state
        if terminated or truncated:
            break
    
    total_rewards = deque()
    gamma = 0.99
    discounted_reward = 0
    # print(rewards)
    for r in rewards[::-1]:
        discounted_reward = r + gamma * discounted_reward
        total_rewards.appendleft(discounted_reward)
    total_rewards = torch.tensor(total_rewards, device=device)
    total_rewards = (total_rewards - total_rewards.mean()) / total_rewards.std()

    states = torch.tensor(np.array(states), device=device)
    logits = policy(states)
    losses = []
    for log_prob, r in zip(saved_log_probs, total_rewards):
        losses.append(-r * log_prob)

    loss = torch.cat(losses).sum()
    print(n_epoch, cum_reward)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # break

env.close()