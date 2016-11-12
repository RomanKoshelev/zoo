from __future__ import print_function

import tensorflow as tf
from core.minds.ddpg.ddpg_mind import DdpgMind
from core.worlds.tentacle_and_apple.tentacle_and_apple_world import TentacleAndAppleWorld


class Platform(object):
    def __init__(self):
        pass

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *args):
        raise NotImplementedError


class TensorflowPlatform(Platform):
    def __init__(self):
        Platform.__init__(self)
        self.session = None

    def __enter__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        return self

    def __exit__(self, *args):
        self.session.close()


# =================================================================================================================

def test_train_world_tentacle_and_apple():
    with TensorflowPlatform() as platform:
        with TentacleAndAppleWorld() as world:
            mind = DdpgMind()
            mind.train(platform, world, episodes=2, steps=100)

# =================================================================================================================

if __name__ == '__main__':
    test_train_world_tentacle_and_apple()
