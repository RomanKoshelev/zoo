from __future__ import print_function

from core.minds.ddpg_mind import DdpgMind
from core.platforms.tensorflow_platform import TensorflowPlatform

from core.worlds.custom_inverted_double_pendulum_world import CustomInvertedDoublePendulumWorld
from core.worlds.inverted_double_pendulum_world import InvertedDoublePendulumWorld
from core.worlds.tentacle_and_apple_world import TentacleAndAppleWorld


# =================================================================================================================

def test_train_world_tentacle_and_apple():
    with TensorflowPlatform() as platform:
        with TentacleAndAppleWorld() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world, episodes=120000, steps=100)


def test_train_world_inverted_double_pendulum():
    with TensorflowPlatform() as platform:
        with InvertedDoublePendulumWorld() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world, episodes=20000)


def test_train_world_custom_inverted_double_pendulum():
    with TensorflowPlatform() as platform:
        with CustomInvertedDoublePendulumWorld() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world, episodes=20000)


def test_train_world_custom_inverted_double_pendulum_with_fat_agent():
    assert False


# =================================================================================================================

if __name__ == '__main__':
    test_train_world_custom_inverted_double_pendulum()
    # test_train_world_inverted_double_pendulum()
    # test_train_world_tentacle_and_apple()
