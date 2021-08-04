#!/usr/bin/env python
# coding: utf-8

# # Play Breakout using the agent

# ## Import Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os                                      # to create folders
import gym                                     # contains the game environment
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from datetime import datetime                  # to print a timestamp
import pickle                                  # to save on file
import torch                                   # ANNs
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# In[2]:


# Crea l'ambiente con il gioco

#env = gym.make('Breakout-v0').unwrapped
env = gym.make('BreakoutDeterministic-v4').unwrapped
#env = gym.make('BreakoutNoFrameskip-v4').unwrapped


# ## Set Up Device
# 
# We import IPython's display module to aid us in plotting images to the screen later.

# In[3]:


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# ## Deep Q-Network

# In[4]:


class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    """
  
    def __init__(self, img_height, img_width, n_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
        def conv2d_size_out(size, kernel_size, stride=1, padding=0):
            return int(size + 2 * padding - kernel_size) // stride  + 1
        
        convw = conv2d_size_out(
                    conv2d_size_out(
                        conv2d_size_out(img_height, kernel_size=8, stride=4
                    ), kernel_size=4, stride=2
                ), kernel_size=3, stride=1)
        convh = conv2d_size_out(
                    conv2d_size_out(
                        conv2d_size_out(img_width, kernel_size=8, stride=4
                    ), kernel_size=4, stride=2
                ), kernel_size=3, stride=1)
        
        linear_input_size = convw * convh * 64  # = 7 * 7 * 64 = 3136
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)


    def forward(self, x):
        """
        Calculates probability of each action.
        Called with either one element to determine next action, or a batch during optimization.
        NOTE: a single discrete state is collection of 4 frames
        :param x: processed state of shape b x 4 x 84 x 84
        :returns tensor of shape [batch_size, n_actions] (estimated action values)
        """
        x = x.to(device)
        x = F.relu(self.conv1(x))  # b x 32 x 20 x 20
        x = F.relu(self.conv2(x))  # b x 64 x 9 x 9
        x = F.relu(self.conv3(x))  # b x 64 x 7 x 7
        x = x.view(x.size(0), -1)  # b x (7 * 7 * 64) x 1
        x = F.relu(self.fc1(x))    # b x 512
        x = self.fc2(x)            # b x  4
        return x


# ## StateHolder class

# In[5]:


class StateHolder:
    """ Class which stores the state of the game.
    We will use 4 consecutive frames of the game stacked together.
    This is necessary for the agent to understand the speed and acceleration of game objects.
    """
    
    def __init__(self, number_screens = 4):
        self.first_action = True
        self.state = torch.ByteTensor(1, 84, 84).to(device)
        self.number_screens = number_screens
        
    def push(self, screen):
        new_screen = screen.squeeze(0)
        if self.first_action:
            self.state[0] = new_screen
            for number in range(self.number_screens-1):
                self.state = torch.cat((self.state, new_screen), 0)
            self.first_action = False
        else:
            self.state = torch.cat((self.state, new_screen), 0)[1:]
    
    def get(self):
        return self.state.unsqueeze(0)

    def reset(self):
        self.first_action = True
        self.state = torch.ByteTensor(1, 84, 84).to(device)


# ## Epsilon Greedy Strategy

# In[6]:


class EpsilonGreedyStrategy():

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, agent_current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * agent_current_step * self.decay)


# ## Reinforcement Learning Agent

# In[7]:


class Agent():

    def __init__(self, strategy, num_actions, device):
        self.strategy     = strategy
        self.num_actions  = num_actions # number of actions that can be taken from a given state
        self.device       = device

    def select_action(self, current_step, state, policy_net):
        rate = self.strategy.get_exploration_rate(current_step)

        if rate > random.random() and state is not None:
            action = random.randrange(self.num_actions)
            return torch.tensor([[action]], device=self.device, dtype=torch.long) # explore      
        else:
            with torch.no_grad():  # since it's not training
                return policy_net(state.float()).argmax(dim=1).to(self.device).view(1, 1) # exploit


# ## Environment Manager

# In[8]:


STATE_W = 84
STATE_H = 84

class EnvManager():

    def __init__(self, env, device):
        self.device = device
        self.env = env
        self.env.reset() # to have an initial observation of the env
        self.max_lives = self.env.ale.lives()
        self.current_screen = None
        self.done = False
        self.n_actions = self.env.action_space.n

    def reset(self):
        """ Resets the env to the initial state
        """
        self.env.reset()
        self.current_screen = None

    def close(self):
        """ Closes the env
        """
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def take_action(self, action):        
        _, reward, self.done, info = self.env.step(action.item())
        return torch.tensor([reward], device=self.device), info

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        """ Returns the current state of the env in the form of a procesed image of the screen
        """
        s = self.get_processed_screen()
        self.current_screen = s
        return s

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render(mode='rgb_array')
        screen = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        # Strip off top and bottom
        return screen[32:195,:]

    def transform_screen_data(self, screen):       
        # Convert to uint, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.uint8).reshape(screen.shape[0],screen.shape[1],1)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((STATE_W, STATE_H)),
            T.ToTensor()
        ])

        return resize(screen).mul(255).type(torch.ByteTensor).to(device).detach().unsqueeze(0) # add a batch dimension (BCHW)


# ## Utility Functions

# ### Plotting

# In[9]:


folder_figs = "figures"
os.makedirs(folder_figs, exist_ok=True)

def plot_durations(values, moving_avg_period, episode):
    plt.figure(1, figsize=(10,5))
    plt.clf()  # Clear the current figure.
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Duration', fontsize=14)
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    filename = os.path.join(folder_figs, "durations_" + version + ".png")
    plt.savefig(filename)
    plt.show()
    print("Episode", episode + len(values), "\n",         moving_avg_period, "episode duration moving avg:", moving_avg[-1])
    #if is_ipython: display.clear_output(wait=True)

def plot_rewards(values, moving_avg_period, episode):
    plt.figure(2, figsize=(10,5))
    plt.clf()
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    filename = os.path.join(folder_figs, "rewards_" + version + ".png")
    plt.savefig(filename)
    plt.show()
    print("Episode", episode + len(values), "\n",         moving_avg_period, "episode reward moving avg:", moving_avg[-1])
    #if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1)             .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
    else:
        moving_avg = torch.zeros(len(values))
    return moving_avg.numpy()

def plot_loss(values):
    plt.figure(3, figsize=(10,5))
    plt.xlabel('Update', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.plot(values)
    filename = os.path.join(folder_figs, "loss_" + version + ".png")
    plt.savefig(filename)
    plt.show()
    if is_ipython: display.clear_output(wait=True)


# ## Main Program

# In[12]:


# Hyperparameters
eps_start           = 1           #
eps_end             = 0.1         # parameters for e-greedy strategy for action selection
eps_decay           = 0.0000001   #


# In[13]:


# Essential Objects

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

em           = EnvManager(env, device)
strategy     = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent        = Agent(strategy, em.n_actions, device)
state_holder = StateHolder()


# In[19]:


# restore checkpoint
version = "01"
checkp_number = 500

folder_save = "models"
folder_checkp = os.path.join(folder_save, "checkpoints_" + version)

filename_checkpoint = os.path.join(folder_checkp, "checkpoint_" + str(checkp_number) + ".pt")
checkpoint = torch.load(filename_checkpoint)

policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.n_actions).to(device)
policy_net.load_state_dict(checkpoint["parameters"])

episode_train   = checkpoint["episode"]
tot_steps_train = checkpoint["tot_steps_done"]


# In[20]:


print("Trained for", episode_train, "episodes. Total steps done:", tot_steps_train)


# In[21]:


filename_durations = "durations.pickle"
filename_rewards   = "rewards.pickle"
filename_losses    = "losses.pickle"

# restore arrays of durations, rewards and losses
filename_durations = os.path.join(folder_checkp, filename_durations)
infile_durations = open(filename_durations, 'rb')
episode_durations = pickle.load(infile_durations)
infile_durations.close()

filename_rewards = os.path.join(folder_checkp, filename_rewards)
infile_rewards = open(filename_rewards, 'rb')
episode_rewards = pickle.load(infile_rewards)
infile_rewards.close()

filename_losses = os.path.join(folder_checkp, filename_losses)
infile_losses = open(filename_losses, 'rb')
losses = pickle.load(infile_losses)
infile_losses.close()


# In[22]:


print(len(episode_durations), len(episode_rewards), len(losses))


# ### Play an episode

# Let's play an episode to see if it learned to play:

# In[28]:


policy_net.eval()

tot_steps_done = tot_steps_train

for episode in range(1):
    em.reset()
    state_holder.push(em.get_state())
    episode_reward = 0
    
    for timestep in count():
        em.render()
        
        state  = state_holder.get()
        action = agent.select_action(tot_steps_done, state, policy_net)
        reward, info = em.take_action(action)
        episode_reward += reward.item()
        
        state_holder.push(em.get_state())
        state = state_holder.get()
        
        tot_steps_done += 1
        
        if em.done:
            print("Reward", episode_reward)
            print("Total steps done", timestep)
            break
        
em.close()

