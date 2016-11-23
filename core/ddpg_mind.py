from alg.ddpg_peter_kovacs.algorithm import DDPG_PeterKovacs
import numpy as np


class DdpgMind:
    def __init__(self, platform, world):
        self.platform = platform
        self.world = world
        self._algorithm = None
        self._reward = None
        self._max_q = None

    def __enter__(self):
        self._algorithm = DDPG_PeterKovacs(self.platform.session, self.world)
        return self

    def __exit__(self, *args):
        pass

    def train(self, weigts_path, episodes, steps):

        self._reward = 0
        self._max_q = []

        def on_step(r, maxq):
            self.world.render()
            self._reward += r
            self._max_q.append(maxq)

        def on_episod(ep):
            save_every_episodes = 100

            max_q = np.mean(self._max_q)  # type: float
            print("Ep: %3d  |  Reward: %+7.0f  |  Qmax: %+7.0f" %
                  (ep, self._reward, max_q))

            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                print("")
                self._algorithm.save(weigts_path)
                print("")

            self.max_q = []
            self._reward = 0

        return self._algorithm.train(episodes, steps, on_episod, on_step)

    def save(self, path):
        self._algorithm.save(path)

    def restore(self, path):
        self._algorithm.restore(path)

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self._algorithm.world_action(a)
