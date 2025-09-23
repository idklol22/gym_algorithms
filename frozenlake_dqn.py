import pandas as pd
import numpy as np 
import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="8x8",is_slippery=False)
state = env.reset()[0]