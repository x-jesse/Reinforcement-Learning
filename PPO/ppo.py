import gymnasium as gym

import torch
import torch.nn as nn

env = gym.make('MountainCarContinuous-v0', render_mode = 'human')
state, info = env.reset()




