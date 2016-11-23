from __future__ import print_function

from core.ddpg_mind import DdpgMind
from core.demo_procedure import DemoProc
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.experiment import Experiment


def demo_mujoco_tentacle_world():
    train = DemoProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind, episodes=30000, steps=150)
    exp = Experiment("001", train)
    exp()

if __name__ == '__main__':
    demo_mujoco_tentacle_world()
