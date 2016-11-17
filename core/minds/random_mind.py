from core.mind import Mind
import numpy as np


class RandomMind(Mind):
    def __init__(self):
        Mind.__init__(self)

    def _train(self, platform, world, episodes, steps):
        raise NotImplementedError

    def _predict(self, world, state):
        return np.random.standard_normal(world.act_dim) * world.act_box
