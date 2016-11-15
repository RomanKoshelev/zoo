from __future__ import print_function

from core.agents.custom_inverted_double_pendulum_agent import CustomInvertedDoublePendulumAgent
from core.minds.ddpg_mind import DdpgMind
from core.platforms.tensorflow_platform import TensorflowPlatform
from core.worlds.custom_inverted_double_pendulum_world import CustomInvertedDoublePendulumWorld
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
        with CustomInvertedDoublePendulumWorld() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world)


def test_train_world_custom_inverted_double_pendulum_with_custom_agent():
    with TensorflowPlatform() as platform:
            agent = CustomInvertedDoublePendulumAgent()
            with CustomInvertedDoublePendulumWorld(agent) as world:
                print(world.summary)
                mind = DdpgMind()
                mind.train(platform, world)


# =================================================================================================================


if __name__ == '__main__':
    test_train_world_custom_inverted_double_pendulum_with_custom_agent()
    # test_train_world_custom_inverted_double_pendulum()
    # test_train_world_tentacle_and_apple()
