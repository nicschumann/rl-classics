import math
import random

from lib.memory import ReplayMemory
from lib.cartpole import reset, step, random_action, render, STATE_SPACE_SIZE, ACTION_SPACE_SIZE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
from tqdm import tqdm


# MDP and RL Hyperparameters

epsilon_init = 1.0
epsilon_end = 0.01

total_episodes = 10000
alpha = 0.001
gamma = 0.95
batch_size = 64


# NN Hyperparameters

HIDDEN_SPACE_SIZE = 24
REPLAY_MEMORY_SIZE = 1000000

class VFA(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(STATE_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense2 = nn.Linear(HIDDEN_SPACE_SIZE, HIDDEN_SPACE_SIZE)
        self.dense3 = nn.Linear(HIDDEN_SPACE_SIZE, ACTION_SPACE_SIZE)

    def forward(self, x):
        y = F.relu( self.dense1(x) )
        y = F.relu( self.dense2(y) )
        return self.dense3(y)

q = VFA()
optimizer = opt.Adam(q.parameters(), lr=alpha)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)

def policy(s, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random_action()
    else:
        return torch.argmax(q(s))

def get_linear_decay(i):
    n = max((total_episodes - i) / total_episodes, 0)
    return (epsilon_init - epsilon_end) * n + epsilon_end

def get_exponential_decay(i):
    epsilon_decay = 100
    n = math.exp(-1 * i / epsilon_decay)
    return (epsilon_init - epsilon_end) * n + epsilon_end




# Testing and Logging Parameters

plot_update_interval = 5
test_interval = 100

per_episode_durations = []
per_episode_losses = []
per_test_durations = []
running_avg_durations = []
epsilon_values = []

fst = lambda t: t[0]
snd = lambda t: t[1]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

def plot():
    ax1.set_title('Episode Durations')
    ax1.plot(list(map(fst, per_episode_durations)), list(map(snd, per_episode_durations)), color="blue", zorder=0)
    ax1.plot(list(map(fst, running_avg_durations)), list(map(snd, running_avg_durations)), color="red", zorder=5)
    ax1.scatter(list(map(fst, per_test_durations)), list(map(snd, per_test_durations)), color="orange", zorder=10)

    ax2.set_title('Loss')
    ax2.plot(list(map(fst, per_episode_losses)), list(map(snd, per_episode_losses)), color="blue")

    ax3.set_title('Epsilon')
    ax3.plot(list(map(fst, epsilon_values)), list(map(snd, epsilon_values)), color="blue")

    plt.pause(0.001)



# Minibatch Prep function

def prepare_minibatch(batch):
    s = torch.cat([b[0] for b in batch]).reshape(-1, STATE_SPACE_SIZE)
    a = F.one_hot(torch.tensor([b[1] for b in batch]).long(), num_classes=ACTION_SPACE_SIZE)

    r = torch.tensor([b[2] for b in batch])

    s_prime = torch.cat([b[3] for b in batch]).reshape(-1, STATE_SPACE_SIZE)
    a_prime = F.one_hot(torch.tensor([b[4] for b in batch]).long(), num_classes=ACTION_SPACE_SIZE)

    # calculate predicted values in batch
    values = torch.sum(q(s) * a, dim=1)

    #calculate the actual td targets we want to optimize for
    td_targets = r + gamma * torch.sum(q(s_prime) * a_prime, dim=1)

    # fix td targets in the terminal state to just be the final reward.
    for i, b in enumerate(batch):
        if b[5]: td_targets[i] = b[2]

    return values, td_targets.detach() # detatch the opt target from the graph


# RL Loop

for i in tqdm(range(total_episodes), desc="q-value function"):

    epsilon = get_exponential_decay(i)
    s, done = reset(), False
    a = policy(s, epsilon)
    duration = 0
    avg_loss = 0

    # collect more experience
    while not done:
        s_prime, r, done = step(a)
        a_prime = policy(s_prime, epsilon)

        sample = (s, a, r, s_prime, a_prime, done) # yes, sarsa
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
        values, td_targets = prepare_minibatch(minibatch)

        loss = F.mse_loss(values, td_targets)

        # do gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = loss.detach().numpy()

        if not done: duration += 1

    scheduler.step()

    # record data
    per_episode_losses.append((i, avg_loss / duration))
    per_episode_durations.append((i, duration))
    epsilon_values.append((i, epsilon))

    # testing and plotting
    if i % test_interval == 0:
        s, done = reset(), False
        test_duration = 0

        while not done:
            render()
            a = policy(s, 0)
            s, r, done = step(a)
            if not done: test_duration += 1

        per_test_durations.append((i, test_duration))

        window = list(map(snd, per_episode_durations[i-test_interval:i]))
        running_avg_durations.append((i, sum(window) / test_interval))

    if i % plot_update_interval == 0:
        plot()
