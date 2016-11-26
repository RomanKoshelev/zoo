from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.demo_procedure import DemoProc
from core.experiment import Experiment
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 1000,
        'steps': 100,
    }

    train_proc = DemoProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind)
    experiment = Experiment("003.demo", train_proc, ini_from="003")

    experiment.start()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
