from enum import Enum, auto

import numpy as np

OKRED = "\033[91m"
OKBLUE = "\033[94m"
ENDC = "\033[0m"
OKGREEN = "\033[92m"
OKYELLOW = "\033[93m"


class TrainTest(Enum):
    train = auto()
    test = auto()


class RouteScoreType(Enum):
    Low = auto()
    Medium = auto()
    High = auto()

class ReplayBuffer:
    def __init__(self, max_size, sample_size):
        self.max_size = max_size
        self.sample_size = sample_size
        self.buffer = [None] * max_size
        self.idx = 0

    def store_item(self, item):
        self.buffer[self.idx] = item
        self.idx += 1 
        self.idx = self.idx % self.max_size

    def store_batch(self, tsp_output, thetas, tsp_data):
        for item in zip(tsp_output, thetas, tsp_data):
            self.store_item(item)

    def sample(self):
        idxs = np.random.randint(0, min(self.max_size, self.idx), self.sample_size)
        data = [None]*self.sample_size
        for counter, idx in enumerate(idxs):
            data[counter] = self.buffer[idx]
        return list(zip(*data))
