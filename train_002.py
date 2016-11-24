from __future__ import print_function

from core.ddpg_mind import DdpgMind
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.train_procedure import TrainProc
from core.experiment import Experiment


def noise_rate_method(ep, eps):
    rmax = .9
    rmin = .1
    rpow = 2.
    k = ep / float(eps)
    return 1 - (rmin + (rmax - rmin) * k ** rpow)


def train_mujoco_tentacle_world():
    train = TrainProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind, episodes=30000, steps=300)

    params = {
        'train.noise.sigma': .1,
        'train.noise.theta': .01,
        'train.noise.rate': .2,
        'train.noise.method': noise_rate_method,
    }

    # train.noise_rate
    # buffer_size
    # batch_size
    # learning_rate
    # target_feedeng_method
    # print experiment sumamry every 100 episodes
    # print params every 100 episodes
    # save params to out_path/params.txt

    exp = Experiment("002", train, params)
    exp()


def test_nois_method():
    rates = []
    eps = 10000
    for ep in range(eps):
        rates.append(noise_rate_method(ep, eps))
    import matplotlib.pyplot as plt
    plt.plot(rates)
    plt.show()


if __name__ == '__main__':
    test_nois_method()
    # train_mujoco_tentacle_world()


