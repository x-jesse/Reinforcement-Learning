import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

class FeedFoward(nn.Module):
    def __init__(self, obs, actions):
        super().__init__()
        self.l1 = nn.Linear(obs, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, actions)
    
    def foward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

env = gym.make('Pendulum-v1', render_mode='human')
print(env.observation_space.shape, env.action_space.shape)
actor = FeedFoward(*env.observation_space.shape, *env.action_space.shape)
critic = FeedFoward(*env.observation_space.shape, *env.action_space.shape)

state, _ = env.reset()

