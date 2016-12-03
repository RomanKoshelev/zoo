import numpy as np


def random_target(target_range):
    x, y, z = np.random.uniform(low=-1, high=+1, size=3) * target_range
    return np.array([x, y, z])
