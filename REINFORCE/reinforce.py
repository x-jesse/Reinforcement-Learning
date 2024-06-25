"""
    ===========================================================
    Welcome to REINFORCE (aka. Monte-Carlo Policy Gradients)!
    ===========================================================

    REINFORCE is probably the equivalent of the "hello world" for reinforcement learning. 
    Though it's not widely used today, it's one of the foundational algorithms used in RL - 
    understanding it is crucial to learning what the more complex algorithms do. The
    implementation is not too long - just <100 lines of code! But there are some nuances that
    require a bit of math to fully understand. 

    This section of the guide will serve as a deep-dive into the concepts and driving principles 
    behind the REINFORCE algorithm. 

    Additional notes:
    - REINFORCE should not be confused with VPG (vanilla policy gradient). Though similar, they
    are slightly different in their calculation of return.
    - This algorithm is the direct implmementation of the pseudo-code from "Intro. to RL" by 
    Richard S. Sutton and Andrew G. Barto. Feel free to check out that book for additional
    explanations!

    Please note: this section does assume you have a basic understanding of Python and Pytorch
    syntax, but it is by no means required to understand the content here.

    Happy learning! <3

"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from collections import deque

"""
Let's start with a description of our environment. The problem we want to solve is simple:
balance a pole on a cart in a 2D linear environment. Our agent will act in one of two ways:
move the cart right, or move the cart left. It acts by applying a fixed force to one side of
the cart. The goal is to keep the pole upright for as long as possible. The agent receives a
reward (capped at 500) for each timestep the pole is upright, and the episode terminates when 
the pole falls below a certain elevation.

You can find detailed documentation here:
https://gymnasium.farama.org/environments/classic_control/cart_pole/

"""

# This is a standard feed-forward neural network :)
# Optional reading: https://en.wikipedia.org/wiki/Feedforward_neural_network
class FeedForward(nn.Module):
    """
    We begin with a simple neural network. This will serve as our policy - 
    a way for us to sample actions. Initially, all the weights will be random;
    our actions will be nonsensical and arbitrary. But as 

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

# we can set our model to run on GPU or Apple Silicon if it's available
# better hardware = faster training
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

# creates the environment
env = gym.make('CartPole-v1', render_mode=None)
state, info = env.reset()

# initializes our policy
policy = FeedForward(env.observation_space.shape[0], env.action_space.n)
policy.to(device)

# our agent should perform pretty well after 200 iterations
# in other words, our policy has converged after 200 episodes
max_eps = 200

# we won't discuss optimizers here - just think of them as a function that performs backpropagation
# Optional reading: 
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

# training loop training loop training loop training loop 
for n_epoch in range(max_eps):
    """
    
    """
    cum_reward = 0
    states, actions, rewards = [], [], []
    saved_log_probs = []
    state, info = env.reset()

    # loop until our episode terminates
    while True:
        # we need to wrap our state in an extra dimension,
        # otherwise PyTorch gets angry when we try to calculate loss later
        state = torch.tensor(state, device=device).unsqueeze(0)
        # gets the logit outputs from our policy
        logits = policy(state)

        # we convert our logits into a categorical probability distribution
        # so we can sample an action from it
        # recall that we want to balance exploration vs exploitation:
        # always selecting the action with the highest probability won't lead to exploration
        action_distr = Categorical(logits=logits)
        action = action_distr.sample()

        # tracks our actions and log probabilities
        actions.append(action)
        saved_log_probs.append(action_distr.log_prob(action))
        
        new_state, reward, terminated, truncated, info = env.step(action.item())
        states.append(new_state)
        rewards.append(reward)
        cum_reward += reward
        
        state = new_state
        if terminated or truncated:
            break
    
    # discount factor hyperparameter - typically set to somewhere around ~1
    # (but not greater than 1)
    gamma = 0.99
    discounted_reward = 0
    total_rewards = deque()

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

env.close()


