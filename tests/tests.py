from __future__ import print_function

from core.minds.ddpg.ddpg_mind import DdpgMind
from core.platforms.tensorflow_platform import TensorflowPlatform


# =================================================================================================================
from core.worlds.inverted_double_pendulum import InvertedDoublePendulum
from core.worlds.tentacle_and_apple import TentacleAndApple


def test_train_world_tentacle_and_apple():
    with TensorflowPlatform() as platform:
        with TentacleAndApple() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world, episodes=120000, steps=100)


def test_train_world_inverted_double_pendulum():
    with TensorflowPlatform() as platform:
        with InvertedDoublePendulum() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world, episodes=20000, steps=None)


# =================================================================================================================

if __name__ == '__main__':
    test_train_world_inverted_double_pendulum()
