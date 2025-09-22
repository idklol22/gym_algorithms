import gymnasium 
import pandas as pd 
import numpy as np 

env = gymnasium.make("Acrobot-v1")
state = env.reset()[0]
epsilon = 1
decay = 0.0006
lr = 0.1
discount = 0.99
q = np.zeros((16,16,16,16,16,16,env.action_space.n))
def state_to_discrete(obs: np.ndarray):
    k = np.empty(6, dtype=int)
    bins_cos = np.linspace(-1.0, 1.0, 17)
    bins_vel = np.linspace(-28.274334, 28.274334, 17)
    bins_vel1 = np.linspace(-12.566371 , 12.566371 , 17)

    for i in range(4):              # cos/sin in [-1,1]
        k[i] = np.digitize(obs[i], bins_cos)
    for i in range(4, 5):           # velocities
        k[i] = np.digitize(obs[i], bins_vel1)
    for i in range(5,6):
        k[i] = np.digitize(obs[i], bins_vel)
    k = np.clip(k, 0, 15)
    return tuple(k)


for i in range(140000):
    reward1 = 0 

    state = env.reset()[0]
    state = state_to_discrete(state)
    truncated = False
    terminated = False

    while not truncated and not terminated:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state])
        newstate, reward, terminated,truncated, _ = env.step(action)
        newstate = state_to_discrete(newstate)
        q[state][action] = q[state][action] + lr * (reward + discount * np.max(q[newstate]) - q[state][action])
        reward1 += reward
        state = newstate
    if epsilon > 0.1:
        epsilon -= decay
    print(reward1)
