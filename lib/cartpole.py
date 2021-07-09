import gym

import torch


env = gym.make('CartPole-v1')

STATE_SPACE_SIZE = 4
ACTION_SPACE_SIZE = 2


def reset():
    return torch.tensor(env.reset()).float()


def step(a):
    s, r, done, _ = env.step( int(a.numpy()) ) # may require a cast from a tensor to numpy

    return torch.tensor(s).float(), torch.tensor(r).float(), done

def random_action():
    return torch.tensor(env.action_space.sample()).float()

def render():
    env.render()
