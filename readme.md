# Solving OpenAI Gymnasium Environments via Reinforcement Learning

Hi :smiley:! This repo is a collection of RL algorithms implemented from scratch using PyTorch with the aim of solving a variety of environments from the Gymnasium library. The purpose is to provide both a theoretical and practical understanding of the principles behind reinforcement learning to someone with little to no experience in machine learning :grinning:. 

## Usage

## Background

Given the complexity and breadth that many reinforcement learning algorithms cover, the topics introduced in this section will provide a high-level coverage of the basic terminology and fundamental concepts used in RL. Detailed explanations of each algorithm can be found in their respective folders. 

### What is RL?

Reinforcement learning, as the name suggests, is the process by which an agent learns through "reinforcing" "good" behaviour. When the agent performs the way we want it to, we provide some quantifiable reward to further encourage this behaviour in the future. If the agent acts undesirably, we "punish" it by providing it with substantially less, or even negative rewards. Just like teaching a poorly-behaved child, we hope that by continually rewarding the agent when it performs well, it will learn to act in a way that is appropriate for our needs.

### The Agent, State, and Environment

RL has several key components that operate in a continuous cycle: the ***agent***, which will perform some action; the ***state***, which represents the current status of the agent relative to its environment; and the ***environment***, which respresents the surrounding world that our agent will act in. In the simple example of a maze navigation robot, the agent would be our robot, the state would represent the current position of the robot in our maze, and the environment would be the maze itself.

<img src="./visuals/rl-cycle.png" width=300>

Each time our agent acts, it may change its state in some way. (Eg. If our robot moves up through maze, its position will change until it hits a wall. If it continues to try to move upwards, it will be impeded by the wall and its state will not change.) Moving to a state has the potential to give some reward. If our robot manages to solve the maze - if it manages to reach the state corresponding to the exit of the maze - we should give it a reward and terminate the process. 

States that result in termination are called *terminal states*. Termination usually means the agent has either successfully solved the task or performed so poorly that it has no way of recovering. Consider an example of teaching an RL agent to play a video game - if our agent dies at any point, it has no way of continuing the game. In other words, it has reached a terminal state.

![animated gif of mario level](./visuals/giphy.webp)

The duration of time from the agent's inception to it reaching a terminal state is called an ***episode***. In the video game example, an episode may correlate to the start of a level until the agent either completes the level or dies in the process. Upon reaching a terminal state, we then place the agent back at the beginning for it to repeat the process again, until it learns how to consistently complete the task. 

### The RL Problem


