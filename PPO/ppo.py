import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gymnasium as gym

class FeedFoward(nn.Module):
    def __init__(self, obs, actions):
        super().__init__()
        self.l1 = nn.Linear(obs, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, actions)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
    

parser = argparse.ArgumentParser()
parser.add_argument('-hu', '--human', action='store_true')
args = parser.parse_args()

env = gym.make('Pendulum-v1', render_mode='human' if args.human else None)
print(env.observation_space.shape, env.action_space.shape)
act_dim = env.action_space.shape[0]
obs_dim = env.action_space.shape[0]
actor = FeedFoward(obs_dim, act_dim)
critic = FeedFoward(obs_dim, act_dim)

covar = torch.diag(torch.full(size=env.action_space.shape, fill_value=0.5))
print(covar)
state, _ = env.reset()

state = torch.tensor(state).unsqueeze(1)
print(state)
print(actor(state))
distr = MultivariateNormal(actor(state), covar)

action = distr.sample()
print(action)