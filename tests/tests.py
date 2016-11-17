from __future__ import print_function

from core.agents.heavy_pendulum_agent import HeavyPendulumAgent
from core.agents.standard_pendulum_agent import StandardPendulumAgent
from core.minds.ddpg_mind import DdpgMind
from core.platforms.tensorflow_platform import TensorflowPlatform
from core.worlds.inverted_double_pendulum_world import InvertedDoublePendulumWorld
from core.worlds.tentacle_and_apple_world import TentacleAndAppleWorld


# =================================================================================================================

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


def test_train_world_custom_inverted_double_pendulum_with_custom_agent():
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


# =================================================================================================================


if __name__ == '__main__':
    test_heavy_pendulum_agent()
    # test_train_world_custom_inverted_double_pendulum_with_custom_agent()
    # test_train_world_custom_inverted_double_pendulum()
    # test_train_world_tentacle_and_apple()
