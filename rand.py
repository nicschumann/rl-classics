import math
import random

from lib.cartpole import reset, step, random_action, render, STATE_SPACE_SIZE, ACTION_SPACE_SIZE

import matplotlib.pyplot as plt
from tqdm import tqdm


# MDP and RL Hyperparameters

total_episodes = 10000


def policy(s):
    return random_action()



# Testing and Logging Parameters

plot_update_interval = 20
test_interval = 100

per_episode_durations = []
per_test_durations = []
running_avg_durations = []

fst = lambda t: t[0]
snd = lambda t: t[1]

fig, ax = plt.subplots()

def plot():
    ax.set_title('Episode Durations')
    ax.plot(list(map(fst, per_episode_durations)), list(map(snd, per_episode_durations)), color="blue", zorder=0)
    ax.plot(list(map(fst, running_avg_durations)), list(map(snd, running_avg_durations)), color="red", zorder=5)
    ax.scatter(list(map(fst, per_test_durations)), list(map(snd, per_test_durations)), color="orange", zorder=10)

    plt.pause(0.001)



# RL Loop

for i in tqdm(range(total_episodes), desc="random agent"):

    s, done = reset(), False
    a = policy(s)
    duration = 0
    avg_loss = 0

    # collect more experience
    while not done:
        s_prime, r, done = step(a)
        a_prime = policy(s_prime)

        s = s_prime
        a = a_prime

        if not done: duration += 1

    # record data
    per_episode_durations.append((i, duration))

    # testing and plotting
    if i % test_interval == 0:
        s, done = reset(), False
        test_duration = 0

        while not done:
            render()
            a = policy(s)
            s, r, done = step(a)
            if not done: test_duration += 1

        per_test_durations.append((i, test_duration))

        window = list(map(snd, per_episode_durations[i-test_interval:i]))
        running_avg_durations.append((i, sum(window) / test_interval))

    if i % plot_update_interval == 0:
        plot()
