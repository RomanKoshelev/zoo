from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.train_procedure import TrainProc
from core.experiment import Experiment


def staircase_noise_rate_5(progress):
    assert 0. <= progress <= 1.

    nr = 1.

    if progress > .01:
        nr = 0.7
    if progress > .05:
        nr = 0.5
    if progress > .4:
        nr = 0.3
    if progress > .6:
        nr = 0.1
    if progress > .9:
        nr = 0.0

    return nr


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 30000,
        'steps': 300,
        'train.noise.sigma': .1,
        'train.noise.theta': .01,
        'train.noise.rate': .2,
        'train.noise.method': staircase_noise_rate_5,
    }

    train = TrainProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind)
    exp = Experiment("001", train)
    exp()


def test_noise_method():
    rates = []
    eps = 30000
    for ep in range(eps):
        rates.append(staircase_noise_rate_5(ep / float(eps)))
    import matplotlib.pyplot as plt
    plt.plot(rates)
    plt.show()


if __name__ == '__main__':
    # test_noise_method()
    train_mujoco_tentacle_world()
