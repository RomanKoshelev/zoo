from __future__ import print_function

from core.context import Context
from tf.algorithm import TensorflowAlgorithm
from utils.ou_noise import OUNoise


class RandomAlgorithm(TensorflowAlgorithm):
    def __init__(self, session, scope, obs_dim, act_dim):
        super(self.__class__, self).__init__(session, scope, obs_dim, act_dim)
        self.noise = OUNoise(act_dim, mu=0,
                             sigma=Context.config['alg.noise_sigma'],
                             theta=Context.config['alg.noise_theta'])

    def predict(self, s):
        return self.noise.noise()

    def train(self, episodes, steps, **callbacks):
        raise NotImplementedError

    def _save_weights(self, path):
        pass

    def _restore_weights(self, path):
        pass
