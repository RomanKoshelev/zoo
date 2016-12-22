from __future__ import print_function

from core.context import Context
from core.tensorflow_algorithm import TensorflowAlgorithm
from utils.ou_noise import OUNoise


class RandomAlgorithm(TensorflowAlgorithm):
    def __init__(self, sess, world):
        super(self.__class__, self).__init__(sess, world.env_id)
        self.noise = OUNoise(world.act_dim, mu=0,
                             sigma=Context.config['alg.noise_sigma'],
                             theta=Context.config['alg.noise_theta'])

    def train(self, episodes, steps, on_episode):
        pass

    # noinspection PyUnusedLocal
    def predict(self, s):
        return self.noise.noise()

    def save_weights(self, path):
        pass

    def restore_weights(self, path):
        pass
