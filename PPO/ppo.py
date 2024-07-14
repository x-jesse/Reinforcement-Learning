import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gymnasium as gym


# This is a standard feed-forward neural network :)
# Optional reading: https://en.wikipedia.org/wiki/Feedforward_neural_network
class FeedForward(nn.Module):
    """
    We begin with a simple PyTorch neural network. This will serve as our policy - 
    a way for us to sample actions. Initially, all the weights will be random;
    our actions will be nonsensical and arbitrary. But as we learn, our policy
    will eventually converge to the optimal one, and our agent should consistently 
    be able to achieve the maximum reward of 500.

    """
    def __init__(self, n_obs, n_actions):
        super().__init__()
        # the number of nodes per layer is pretty arbitrary
        # as long as the network is sufficiently complex to solve our problem
        # it doesn't really matter how many nodes we have
        # (though simpler networks will usually train faster)
        self.l1 = nn.Linear(n_obs, 64)
        self.l2 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        # ReLU activation
        # Optional reading: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        x = F.relu(self.l1(x))
        return self.l2(x)
    

parser = argparse.ArgumentParser()
parser.add_argument('-hu', '--human', action='store_true')
args = parser.parse_args()

env = gym.make('Pendulum-v1', render_mode='human' if args.human else None)
state, _ = env.reset()
print(env.observation_space.shape, env.action_space.shape)
act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
actor = FeedForward(obs_dim, act_dim)

critic = FeedForward(obs_dim, act_dim)

covar = torch.diag(torch.full(size=env.action_space.shape, fill_value=0.5))
# print(covar)

# gather trajectories
"""
    We set a hard limit for the data collection portion of our policy, limiting it to 200 timesteps.
    This also means that the policy must learn to complete the task in under 200 timesteps.
    Coincidentally, Gymnasium will also truncate the episode at 200 timesteps, so we technically don't need this 
    condition. I feel that it's more clear to set it explicitly though.

"""
max_timesteps = 200
for tstep in range(max_timesteps):
    state = torch.tensor(state)
    distr = MultivariateNormal(actor(state), covar)
    action = distr.sample()

    new_state, reward, terminated, truncated, _ = env.step(action.numpy())

    if terminated or truncated:
        break

    state = new_state