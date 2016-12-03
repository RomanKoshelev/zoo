import numpy as np

from core.context import Context


def random_target():
    x, z = np.random.uniform(low=-1, high=+1, size=2) * Context.config['env.target_range_xz']
    return np.array([x, z])
