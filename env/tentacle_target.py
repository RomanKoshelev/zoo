import numpy as np


def random_target(target_range):
    x, z = np.random.uniform(low=-1, high=+1, size=2) * target_range
    return np.array([x, z])
