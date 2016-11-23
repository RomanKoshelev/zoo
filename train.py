from __future__ import print_function

from core.tentacle_agent import TentacleAgent
from core.ddpg_mind import DdpgMind
from core.tensorflow_platform import TensorflowPlatform
from core.standard_proc import StandardProc
from core.tentacle_world import TentacleWorld


class Experiment:
    def __init__(self):
        pass


def train_mujoco_tentacle_world():
    proc = StandardProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind)

    # exp = Experiment(proc, "./out/", "My first experiment")

    proc.train("./out/DDPG_PeterKovacs_Zoo_Mujoco_Tentacle_v1/", episodes=10000, steps=150)


if __name__ == '__main__':
    train_mujoco_tentacle_world()
