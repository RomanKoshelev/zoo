from __future__ import print_function

from core.minds.ddpg.ddpg_mind import DdpgMind
from core.worlds.tentacle_and_apple.tentacle_and_apple_world import TentacleAndAppleWorld
from core.platforms.tensorflow_platform import TensorflowPlatform


# =================================================================================================================

def test_train_world_tentacle_and_apple():
    with TensorflowPlatform() as platform:
        with TentacleAndAppleWorld() as world:
            print(world.summary)
            mind = DdpgMind()
            mind.train(platform, world, episodes=120000, steps=100)


# =================================================================================================================

if __name__ == '__main__':
    test_train_world_tentacle_and_apple()
