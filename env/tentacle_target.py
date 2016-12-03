import numpy as np


def random_target(tentacle_env):
    x, z = np.random.uniform(+1, +1, 2) * tentacle_env.target_range
    return np.array([x, z])
