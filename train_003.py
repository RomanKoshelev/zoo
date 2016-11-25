from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.train_procedure import TrainProc
from utils.noise_tools import staircase_5
import os


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 10000,
        'steps': 75,
        'train.save_every_episodes': 100,
        'train.batch_size': 640,
        'train.buffer_size': 1e6,
        'train.noise_sigma': .1,
        'train.noise_theta': .01,
        'train.noise_rate_method': staircase_5,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind)
    experiment = Experiment("003", train_proc)

    if os.path.exists( experiment.work_path):
        experiment.proceed()
    else:
        experiment.start()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
