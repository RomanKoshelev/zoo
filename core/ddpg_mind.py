from alg.ddpg_peter_kovacs.algorithm import DDPG_PeterKovacs
import numpy as np


class DdpgMind:
    def __init__(self, platform, world):
        self.platform = platform
        self.world = world
        self._algorithm = None
        self._reward = None
        self._max_q = None
        self._noise_rate = None

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
            print("Ep: %3d  |  Nr: %.0f%%  |  Reward: %+7.0f  |  Qmax: %+7.0f" %
                  (ep, self._noise_rate * 100, self._reward, max_q))

            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                print("")
                self._algorithm.save(weigts_path)
                print("")

            self.max_q = []
            self._reward = 0

        def noise_method(action, noise, episode, _episodes):
            nr_max = 0.5
            nr_min = 0.4
            nr_eps = min(1000., _episodes / 10.)
            nk = 1 - min(1., float(episode) / nr_eps)
            k = nr_min + nk * (nr_max - nr_min)
            self._noise_rate = k
            return (1 - k) * action + k * noise

        return self._algorithm.train(episodes, steps, noise_method, on_episod, on_step)

    def save(self, path):
        self._algorithm.save(path)

    def restore(self, path):
        self._algorithm.restore(path)

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self._algorithm.world_action(a)

