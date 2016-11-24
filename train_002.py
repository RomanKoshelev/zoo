from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.train_procedure import TrainProc
from utils.noise_tools import linear_2


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 30000,
        'steps': 300,
        'train.buffer_size': 1e6,
        'train.noise_sigma': .1,
        'train.noise_theta': .01,
        'train.noise_rate_method': linear_2,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind)
    experiment = Experiment("002", train_proc)
    experiment()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
