import time

import gym
env = gym.make('Taxi-v3')
print(env.reset())
print(env.action_space.n)


