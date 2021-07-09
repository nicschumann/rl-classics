import random

class ReplayMemory():
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []

    def append(self, s):
        if len(self.samples) >= self.max_size: self.samples.pop(0)
        self.samples.append(s)

    def sample(self, batch_size):
        return random.sample(self.samples, min(len(self.samples), batch_size))
