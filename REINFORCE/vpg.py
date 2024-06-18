import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.l1 = nn.Linear(n_obs, 8)
        self.l2 = nn.Linear(8, 8)
        self.l3 = nn.Linear(8, n_actions)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


env = gym.make('MountainCarContinuous-v0', render_mode = 'human')
state, info = env.reset()




