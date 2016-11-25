from __future__ import print_function
from alg.ddpg_peter_kovacs.ddpg import DDPG_PeterKovacs
import numpy as np

from core.context import Context


class DdpgMind:
    def __init__(self, platform, world):
        self.platform = platform
        self.world = world
        self._algorithm = None
        self._reward = None
        self._max_q = None
        self._nr = None

    def __enter__(self):
        self._algorithm = DDPG_PeterKovacs(self.platform.session, self.world)
        return self

    def __exit__(self, *args):
        pass

    def train(self, weigts_path):
        episodes = Context.config['episodes']
        steps = Context.config['steps']
        save_every_episodes = Context.config['save_every_episodes']

        self._reward = 0
        self._max_q = []

        def on_step(r, maxq, nr):
            self.world.render()
            self._reward += r
            self._nr = nr
            self._max_q.append(maxq)
            Context.window_title['step'] = ""

        def on_episod(ep):
            max_q = np.mean(self._max_q)  # type: float
            print("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (ep, self._nr, self._reward, max_q))
            Context.window_title['episod'] = "|  %d/%d: R = %+.0f, Q = %+.0f" % (ep, episodes, self._reward, max_q)

            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                self.save(weigts_path)

            self.max_q = []
            self._reward = 0

        return self._algorithm.train(episodes, steps, on_episod, on_step)

    def save(self, folder):
        print("\nSaving [%s].." % folder)
        self._algorithm.save(self.weights_path(folder))
        print("Done.\n")

    def restore(self, folder):
        print("\nRestoring [%s].." % folder)
        self._algorithm.restore(self.weights_path(folder))
        print("Done.\n")

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self.world.scale_action(a)

    @staticmethod
    def weights_path(folder):
        import os
        return os.path.join(folder, "weights.ckpt")
