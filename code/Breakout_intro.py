#!/usr/bin/env python
# coding: utf-8

# # Breakout

# https://gym.openai.com/envs/Breakout-v0/
# 
# Maximize your score in the Atari 2600 game Breakout. In this environment, the observation is an RGB image of the screen, which is an array of shape $(210, 160, 3)$. Each action is repeatedly performed for a duration of $k$ frames, where $k$ is uniformly sampled from $\{2, 3, 4\}$.

# ## Import Libraries

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# ## Understanding the game environment

# Let's run an instance of the environment from Gym for time steps and will take a random action at each time step. We'll render the environment at each step so we can see what this will look like.
# 
# In this [page](https://github.com/openai/gym/blob/a5a6ae6bc0a5cfc0ff1ce9be723d59593c165022/gym/envs/__init__.py) we can see that the environments for the Atari games can be 

# In[2]:


#env = gym.make('Breakout-v0')
env = gym.make('BreakoutDeterministic-v4')
#env = gym.make('BreakoutNoFrameskip-v4')

env.reset()

for _ in range(3000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()


# In[3]:


print(env.action_space)
print(env.observation_space)


# The `Discrete` space allows a fixed range of non-negative numbers, so in this case valid `action`s are 0 or 1 or 2 or 3. The `Box` space represents an `n`-dimensional box, so valid `observations` will be ????.

# In[10]:


help(env.unwrapped)


# In[4]:


env.unwrapped.get_action_meanings()


# In[11]:


env.unwrapped.get_keys_to_action()


# We can also check the `Box`'s bounds:

# In[5]:


print(env.observation_space.high)
print(env.observation_space.low)


# In[6]:


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(f"observation: {observation}")
        action = env.action_space.sample()
        print(f"action: {action}")
        observation, reward, done, info = env.step(action)
        print(f"reward: {reward}")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

