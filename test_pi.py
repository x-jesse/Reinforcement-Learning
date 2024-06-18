from gridworld import GridWorld
from policy_iteration import PolicyIteration
from tabular_policy import TabularPolicy

gridworld = GridWorld()
policy = TabularPolicy(default_action=gridworld.LEFT)
PolicyIteration(gridworld, policy).policy_iteration(max_iterations=100)
gridworld.visualise_policy(policy)