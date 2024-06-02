import gymnasium as gym
import math
import numpy as np
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

TransitionState = namedtuple('TransitionState', 
                             ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(TransitionState(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super().__init__()
        self.input_layer = nn.Linear(n_observations, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)
    
    def forward(self, x): # forwards propagation
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        return self.output_layer(x)


test_nn = DQN(4, 2)
test_input = torch.tensor([[[1., 1., 1., 1.]], [[1., 1., 1., 1.]]])
print(test_input)
print(test_nn(test_input))