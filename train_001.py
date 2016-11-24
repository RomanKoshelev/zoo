from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.train_procedure import TrainProc
from utils.noise_tools import staircase5_noise_rate


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 30000,
        'steps': 300,
        'train.noise_sigma': .1,
        'train.noise_theta': .01,
        'train.noise_rate_method': staircase5_noise_rate,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind)
    experiment = Experiment("001", train_proc)
    experiment()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
