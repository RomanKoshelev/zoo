from __future__ import print_function

from core.agents.tentacle_agent import TentacleAgent
from core.minds.ddpg_mind import DdpgMind
from core.minds.random_mind import RandomMind
from core.platforms.tensorflow_platform import TensorflowPlatform
from core.procs.standard_proc import StandardProc
from core.worlds.tentacle_world import TentacleWorld


# =================================================================================================================


def test_random_mind():
    with TensorflowPlatform() as platform:
        agent = TentacleAgent()
        with TentacleWorld(agent) as world:
            mind = RandomMind()
            proc = StandardProc()
            proc.demo(platform, world, mind, steps=100)


def test_mujoco_tentacle_world():
    with TensorflowPlatform() as platform:
        agent = TentacleAgent()
        with TentacleWorld(agent) as world:
            mind = RandomMind()
            proc = StandardProc()
            proc.demo(platform, world, mind, steps=300)


def test_train_mujoco_tentacle_world():
    with TensorflowPlatform() as platform:
        agent = TentacleAgent()
        with TentacleWorld(agent) as world:
            mind = DdpgMind()
            proc = StandardProc()
            proc.train(platform, world, mind, steps=300)


# =================================================================================================================


if __name__ == '__main__':
    test_train_mujoco_tentacle_world()
    # test_mujoco_tentacle_world()
