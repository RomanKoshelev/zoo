from __future__ import print_function

import numpy as np
from tf.algorithm import TensorflowAlgorithm


class DummyAlgorithm(TensorflowAlgorithm):
    def __init__(self, session, scope, obs_dim, act_dim):
        super(self.__class__, self).__init__(session, scope, obs_dim, act_dim)

    def predict(self, s):
        return np.zeros(shape=[self._act_dim])

    def train(self, episodes, steps, **callbacks):
        raise NotImplementedError

    def _save_weights(self, path):
        pass

    def can_restore(self, path):
        return True

    def _restore_weights(self, path):
        pass
