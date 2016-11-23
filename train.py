from __future__ import print_function

from core.ddpg_mind import DdpgMind
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.train_procedure import TrainProc
from core.experiment import Experiment


def train_mujoco_tentacle_world():
    proc = TrainProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind, episodes=10000, steps=150)
    exp = Experiment(proc, "001", "My first experiment")
    exp()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
