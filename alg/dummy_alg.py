from __future__ import print_function

import numpy as np
from core.tensorflow_algorithm import TensorflowAlgorithm


class DummyAlgorithm(TensorflowAlgorithm):
    def __init__(self, sess, world):
        super(self.__class__, self).__init__(sess, world.env_id)

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def predict(self, s):
        return np.array([])

    def train(self, episodes, steps, on_episode):
        pass

    def save_weights(self, path):
        pass

    def restore_weights(self, path):
        pass
