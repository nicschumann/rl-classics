import math
import random

from lib.memory import ReplayMemory
from lib.cartpole import reset, step, random_action, render, STATE_SPACE_SIZE, ACTION_SPACE_SIZE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
from tqdm import tqdm


episodes = 1000
gamma = 0.99
planning_depth = 3 # d in the paper
prediction_steps = 5 # k in the paper
# action_search_width = 2 # b in the paper; only 2 actions in this MDP...


# Source: https://arxiv.org/pdf/1707.03497.pdf

INTERNAL_STATE_SIZE = 2
HIDDEN_SPACE_SIZE = 12

# Maps from observations x_t to hidden states s_t
class Encoding(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(STATE_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense2 = nn.Linear(HIDDEN_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense3 = nn.Linear(HIDDEN_SPACE_SIZE, INTERNAL_STATE_SIZE)

    def forward(self, x):
        s = F.relu(self.dense1(x))
        s = F.relu(self.dense2(s))
        s = self.dense3(s)

        return s

# Maps from hidden states s_t, to values V(s)
class Value(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(INTERNAL_STATE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense2 = nn.Linear(HIDDEN_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense3 = nn.Linear(HIDDEN_SPACE_SIZE, 1)

    def forward(self, s):
        v = F.relu(self.dense1(s))
        v = F.relu(self.dense2(v))
        v = self.dense3(v)

        return v

# Maps from hidden state s_t and option to predicted reward and discount factor gamma
class Outcome(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense_state = nn.Linear(INTERNAL_STATE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense_option = nn.Linear(ACTION_SPACE_SIZE, HIDDEN_SPACE_SIZE)

        self.dense2 = nn.Linear(2 * HIDDEN_SPACE_SIZE, HIDDEN_SPACE_SIZE)

        self.dense_reward = nn.Linear(HIDDEN_SPACE_SIZE, 1)

    def forward(self, s, o):
        y_s = F.relu(self.dense_state(s))
        y_o = F.relu(self.dense_option(o))

        y = torch.cat((y_s, y_o), dim=0)
        y = F.relu(self.dense2(y))
        r = self.dense_reward(y)

        return r


# Maps from hidden states s_t and options o_t to new hidden states s_{t+1}
class Transition(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense_state = nn.Linear(INTERNAL_STATE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense_option = nn.Linear(ACTION_SPACE_SIZE, HIDDEN_SPACE_SIZE)

        self.dense2 = nn.Linear(2 * HIDDEN_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense3 = nn.Linear(HIDDEN_SPACE_SIZE, INTERNAL_STATE_SIZE)

    def forward(self, s, o):
        y_s = F.relu(self.dense_state(s))
        y_o = F.relu(self.dense_option(o))
        y = torch.cat((y_s, y_o), dim=0)

        s_prime = F.relu(self.dense2(y))
        s_prime = self.dense3(s_prime)

        return s_prime


f_enc = Encoding()
f_value = Value()
f_out = Outcome()
f_trans = Transition()

f_enc_opt = Adam(f_enc.parameters(), lr=0.001)
f_value_opt = Adam(f_value.parameters(), lr=0.001)
f_out_opt = Adam(f_out.parameters(), lr=0.001)
f_trans_opt = Adam(f_trans.parameters(), lr=0.001)


def q_plan(s, a, steps):
    r = f_out(s, a)
    s_prime = f_trans(s, a)
    v = f_value(s_prime)

    if steps == 1:
        return r + gamma * v
    else:
        q_values = []
        for a_index in range(ACTION_SPACE_SIZE):
            a_prime = F.one_hot(torch.tensor(a_index).long(), num_classes=ACTION_SPACE_SIZE).float()
            q_value = q_plan(s_prime, a_prime, steps - 1)
            q_values.append(q_value)

        q_values = torch.cat(q_values)
        max_q_value = torch.max(q_values, 0, keepdim=True)[0]

        return r + gamma * ( 1/steps * v + (steps - 1) / steps * max_q_value)



def q_policy(x, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random_action()
    else:
        s = f_enc(x)
        values = []

        for a_index in range(ACTION_SPACE_SIZE):
            a = F.one_hot(torch.tensor(a_index).long(), num_classes=ACTION_SPACE_SIZE).float()
            v = q_plan(s, a, planning_depth)
            values.append(v)

        values = torch.cat(values)
        return torch.argmax(values), s







# debug stuff


# / debug stuff


for i in tqdm(range(episodes), desc="running episodes"):
    trajectory = []
    epsilon = 0.01

    x, done, r = reset(), False, 0
    a, s = q_policy(x, epsilon)

    while not done:
        x_prime, r, done = step(a)
        a_prime, s_prime = q_policy(x_prime, epsilon)

        # need to add the prediction data from the policy, so we can bootstrap.

        trajectory.append((x, s, a, r, x_prime, s_prime, a_prime)) # add a sample to the k-step trajectory

    G = r # we are now in a terminal state, and the reward is the last r

    # time to plan
    # go backward down the trajectory
    # TODO: pick up from here: https://arxiv.org/pdf/1707.03497.pdf
    # page 15 / 16
    k_step_r_losses = []
    k_step_value_losses = []

    T = len(trajectory)

    for (x, a, r, x_prime, a_prime) in reversed(trajectory): # reversed trajectory
        G = r + gamma * G # target return
        # https://arxiv.org/pdf/1707.03497.pdf Check page 4 for the definition of the terms in the loss









    # generate trajectories by following the q_planning strategy
    # for the VPN
