import gymnasium as gym
import math
import numpy as np
import random
import sys
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

def select_action(state):
    global steps_done
    sample = random.random()
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_network(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([env.action_space.sample()], dtype=torch.long).unsqueeze(0)
    
def optimize():
    # print("OPTIMIZE START")
    experiences = memorybuffer.sample(BATCH_SIZE)
    batch = TransitionState(*zip(*experiences))
    # print("Training sample:", batch)

    non_final_mask = torch.tensor(tuple(map(lambda state : state is not None, batch.next_state)))

    states = torch.cat(batch.state)
    next_states = torch.cat([s for s in batch.next_state if s is not None])
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    q_values = policy_network(states).gather(1, actions)
    # print("Predicted Q-val:", q_values)
    target_q_values = torch.zeros(BATCH_SIZE) 
    with torch.no_grad():
        target_q_values[non_final_mask] = target_network(next_states).max(1).values
    # print("Target Q-val:", target_q_values)
    # print(target_q_values.max(1))
    
    expected_q_values = (target_q_values * GAMMA + rewards).unsqueeze(1) # discount factor
    # print("Expected q:", expected_q_values)
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()

def main():
    max_reward = 0
    for ep in range(NUM_EPISODES):
        print("Episode num:", ep, "| Max reward achieved:", max_reward, end='\r')

        state, info = env.reset()
        state = torch.tensor(state).unsqueeze(0)
        duration, total_reward = 0, 0
        while True:
            # print("Current state:",  state)
            action = select_action(state)
            # print("Action:", action)

            observation, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation).unsqueeze(0)
            # print("New state:", next_state)
            memorybuffer.push(state, action, next_state, reward)
            # print("Added to memory:", {"State": state, "Action": action, "Next state": next_state, "Reward": reward})

            state = next_state

            if len(memorybuffer) > BATCH_SIZE:
                optimize()
            
            # if ep % 5 == 0:
            #     target_network.load_state_dict(policy_network.state_dict())
            
            target_net_state_dict = target_network.state_dict()
            policy_net_state_dict = policy_network.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_network.load_state_dict(target_net_state_dict)

            if done:
                # print("Is done")
                max_reward = max(max_reward, total_reward)
                break
            
            # print("ITERATION DONE\n")

if __name__ == '__main__':
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 1000
    TAU = 0.005
    LEARNING_RATE = 1e-4

    NUM_EPISODES = 600
    steps_done = 0

    env = gym.make("Humanoid-v4", render_mode="human")
    state, info = env.reset()
    num_obs = len(state)
    num_actions = env.action_space.n

    policy_network = DQN(num_obs, num_actions)
    target_network = DQN(num_obs, num_actions)
    target_network.load_state_dict(policy_network.state_dict())

    optimizer = optim.AdamW(policy_network.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memorybuffer = ReplayBuffer(10000)

    main()