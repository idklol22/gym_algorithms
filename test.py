import gymnasium as gym
import pandas as pd 
import numpy as np
import random
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1
decay = 0.00001
rng = np.random.default_rng(42)
rng 
print(rng)
env = gym.make("FrozenLake-v1", map_name="8x8",is_slippery=False,render_mode="human")
state =   env.reset()[0]
terminated = False
truncated = False
for i in range(400):
    state =   env.reset()[0]
    while not (terminated or truncated):
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
    terminated = False
    truncated = False    
env.close()

