from __future__ import print_function

from core.agents.heavy_pendulum_agent import HeavyPendulumAgent
from core.agents.standard_pendulum_agent import StandardPendulumAgent
from core.agents.tentacle_agent import TentacleAgent
from core.minds.ddpg_mind import DdpgMind
from core.minds.random_mind import RandomMind
from core.platforms.tensorflow_platform import TensorflowPlatform
from core.worlds.inverted_double_pendulum_world import InvertedDoublePendulumWorld
from core.worlds.tentacle_and_apple_world import TentacleAndAppleWorld
from core.procs.standard_proc import StandardProc


# =================================================================================================================
from core.worlds.tentacle_world import TentacleWorld


def test_train_world_tentacle_and_apple():
    with TensorflowPlatform() as platform:
        with TentacleAndAppleWorld() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world, steps=100)


def test_train_world_custom_inverted_double_pendulum():
    with TensorflowPlatform() as platform:
        with InvertedDoublePendulumWorld() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world)


def test_standard_pendulum_agent():
    with TensorflowPlatform() as platform:
            agent = StandardPendulumAgent()
            with InvertedDoublePendulumWorld(agent) as world:
                print(world.summary)
                mind = DdpgMind()
                mind.train(platform, world)


def test_heavy_pendulum_agent():
    with TensorflowPlatform() as platform:
            agent = HeavyPendulumAgent()
            with InvertedDoublePendulumWorld(agent) as world:
                print(world.summary)
                mind = DdpgMind()
                mind.train(platform, world)


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
            proc.demo(platform, world, mind, steps=100)


# =================================================================================================================


if __name__ == '__main__':
    test_mujoco_tentacle_world()
    # test_heavy_pendulum_agent()
    # test_standard_pendulum_agent()
    # test_train_world_custom_inverted_double_pendulum()
    # test_train_world_tentacle_and_apple()
