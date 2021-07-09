import math
import random

from lib.memory import ReplayMemory
from lib.cartpole import reset, step, random_action, render, STATE_SPACE_SIZE, ACTION_SPACE_SIZE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from tqdm import tqdm


# MDP and RL Hyperparameters
total_episodes = 10000
alpha = 0.001
beta = 0.001
gamma = 0.997
batch_size = 128


# NN Hyperparameters

HIDDEN_SPACE_SIZE = 24
REPLAY_MEMORY_SIZE = 1000000

class VFA(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(STATE_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense2 = nn.Linear(HIDDEN_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense3 = nn.Linear(HIDDEN_SPACE_SIZE, 1)

    def forward(self, x):
        y = F.relu( self.dense1(x) )
        y = F.relu( self.dense2(y) )
        return self.dense3(y)

class PGE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(STATE_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense2 = nn.Linear(HIDDEN_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense3 = nn.Linear(HIDDEN_SPACE_SIZE, ACTION_SPACE_SIZE)

    def forward(self, x):
        y = F.relu( self.dense1(x) )
        y = F.relu( self.dense2(y) )
        y = self.dense3(y)
        dist = Categorical(F.softmax(y, dim=-1))
        return dist


v = VFA()
v_opt = opt.Adam(v.parameters(), lr=alpha)
v_sched = StepLR(v_opt, step_size=100, gamma=0.9)

pi = PGE()
pi_opt = opt.Adam(pi.parameters(), lr=beta)
pi_sched = StepLR(pi_opt, step_size=100, gamma=0.95)

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)



# Testing and Logging Parameters

plot_update_interval = 5
test_interval = 100

per_episode_durations = []
per_episode_v_losses = []
per_episode_pi_losses = []
per_test_durations = []
running_avg_durations = []
epsilon_values = []

fst = lambda t: t[0]
snd = lambda t: t[1]

fig, (ax1, ax2) = plt.subplots(1, 2)

def plot():
    ax1.set_title('Episode Durations')
    ax1.plot(list(map(fst, per_episode_durations)), list(map(snd, per_episode_durations)), color="blue", zorder=0)
    ax1.plot(list(map(fst, running_avg_durations)), list(map(snd, running_avg_durations)), color="red", zorder=5)
    ax1.scatter(list(map(fst, per_test_durations)), list(map(snd, per_test_durations)), color="orange", zorder=10)

    ax2.set_title('Losses')
    ax2.plot(list(map(fst, per_episode_v_losses)), list(map(snd, per_episode_v_losses)), color="blue")
    ax2.plot(list(map(fst, per_episode_pi_losses)), list(map(snd, per_episode_pi_losses)), color="red")

    plt.pause(0.001)



# Minibatch Prep function

def prepare_minibatch(batch):
    s = torch.cat([b[0] for b in batch]).reshape(-1, STATE_SPACE_SIZE)

    a = torch.tensor([b[1] for b in batch]).long()

    r = torch.tensor([b[2] for b in batch]).reshape(-1, 1)

    s_prime = torch.cat([b[3] for b in batch]).reshape(-1, STATE_SPACE_SIZE)

    # calculate predicted values in batch
    values = v(s)

    probs = pi(s)
    log_probs = probs.log_prob(a)

    #calculate the actual td targets we want to optimize for
    td_targets = r + gamma * v(s_prime)

    # fix td targets in the terminal state to just be the final reward.
    for i, b in enumerate(batch):
        if b[5]: td_targets[i] = b[2]

    return s, log_probs, values, td_targets.detach() # detatch the opt target from the graph


# RL Loop

for i in tqdm(range(total_episodes), desc="policy gradient"):

    s, done = reset(), False
    probs = pi(s)
    a = probs.sample()

    duration = 0
    avg_v_loss = 0
    avg_pi_loss = 0

    # collect more experience
    while not done:
        s_prime, r, done = step(a)

        probs_prime = pi(s_prime)
        a_prime = probs_prime.sample()

        sample = (s, a, r, s_prime, a_prime, done) # yes, sarsa with log_probs
        replay_memory.append(sample)

        s = s_prime
        a = a_prime

        # do minibatch
        # NOTE: for a good experiment, consider moving the
        # minibatch code outside of the episode loop. In other words,
        # collect an episode of experience without bootstrapping,
        # then do gradient descent. Notice HOW MUCH WORSE policy improvement is
        # ie (for decent Hyperparameters, the thing is solved in 400 episodes).
        minibatch = replay_memory.sample(batch_size)
        states, log_probs, values, td_targets = prepare_minibatch(minibatch)

        # print(values.size())
        # print(td_targets.size())

        advantage = td_targets - values

        pi_loss = -(log_probs * advantage.detach()).mean()
        v_loss = advantage.pow(2).mean()

        v_opt.zero_grad()
        pi_opt.zero_grad()

        pi_loss.backward()
        v_loss.backward()

        pi_opt.step()
        v_opt.step()

        avg_v_loss = v_loss.detach().numpy()
        avg_pi_loss = pi_loss.detach().numpy()

        if not done: duration += 1

    v_sched.step()
    pi_sched.step()

    # record data
    per_episode_v_losses.append((i, avg_v_loss / duration))
    per_episode_pi_losses.append((i, avg_pi_loss / duration))
    per_episode_durations.append((i, duration))

    # testing and plotting
    if i % test_interval == 0:
        s, done = reset(), False
        test_duration = 0

        while not done:
            render()
            probs = pi(s)
            a = probs.sample()
            s, r, done = step(a)
            if not done: test_duration += 1

        per_test_durations.append((i, test_duration))

        window = list(map(snd, per_episode_durations[i-test_interval:i]))
        running_avg_durations.append((i, sum(window) / test_interval))

    if i % plot_update_interval == 0:
        plot()
