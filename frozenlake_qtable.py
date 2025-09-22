import gymnasium as gym
import pandas as pd 
import numpy as np
import random
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1
decay = 0.00006
env = gym.make("FrozenLake-v1", map_name="8x8",is_slippery=False)
state =   env.reset()[0]
q = np.zeros((env.observation_space.n,env.action_space.n))
terminated = False
truncated = False
for i in range(400000):


    while not (terminated or truncated):
        if np.random.rand()<epsilon:
            action = env.action_space.sample()

        else :
            action=q[state].argmax()
        new_state, reward, terminated, truncated, info = env.step(action)
        q[state,action] = q[state,action] + learning_rate*(reward+discount_factor*q[new_state].max()-q[state,action])

        state = new_state
    if epsilon>0.01:
        epsilon = epsilon - decay
    terminated = False
    truncated = False   
    state =   env.reset()[0]
    print(reward)
env.close()

