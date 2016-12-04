from __future__ import print_function

import os

from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from core.context import Context
from utils.string_tools import tab
from utils.os_tools import make_dir_if_not_exists

NETWORK_WEIGHTS_PATH = "network/weights.ckpt"
ALGORITHM_STATE_PATH = "algorithm/state.pickle"


class DdpgMind:
    def __init__(self, platform, world, reporter):
        self.platform = platform
        self.world = world
        self._reporter = reporter
        self._algorithm = None
        self._saved_episode = None

    def __enter__(self):
        self._algorithm = DDPG_PeterKovacs(self.platform.session, self.world)
        return self

    def __exit__(self, *args):
        pass

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "algorithm: " + tab(self._algorithm),
        )

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self.world.scale_action(a)

    def train(self, work_path):
        def on_episod(ep, reward, nr, maxq):
            self._reporter.on_train_episode(ep, nr, reward, maxq)
            self._save_results_if_need(work_path, ep,
                                       Context.config['exp.episodes'],
                                       Context.config['exp.save_every_episodes'])
            self._evaluate_if_need(ep,
                                   Context.config['mind.evaluate_every_episodes'],
                                   Context.config['exp.steps'])

        return self._algorithm.train(Context.config['exp.episodes'], Context.config['exp.steps'], on_episod)

    def _evaluate_if_need(self, ep, evs, steps):
        if (ep + 1) % evs == 0:
            self._reporter.on_evaluiation_start()
            reward = self.run_episode(steps)
            self._reporter.on_evaluiation_end(ep, reward)

    def run_episode(self, steps):
        s = self.world.reset()
        reward = 0
        for t in xrange(steps):
            self.world.render()
            a = self.predict(s)
            s, r, done, _ = self.world.step(a)
            reward += r
        return reward

    def save(self, folder):
        self._reporter.log('Saving weights and algorithm state...')
        self.save_weights(folder)
        self.save_algorithm_state(folder)

    def restore(self, folder):
        self.restore_weights(folder)
        self.restore_algorithm_state(folder)

    def restore_weights(self, folder):
        self._algorithm.restore_weights(self.network_weights_path(folder))

    def save_weights(self, folder):
        self._algorithm.save_weights(make_dir_if_not_exists(self.network_weights_path(folder)))

    def save_algorithm_state(self, folder):
        import pickle
        path = self.algorithm_state_path(folder)
        make_dir_if_not_exists(path)
        with open(path, 'w') as f:
            pickle.dump([
                self._algorithm.episode,
                self._algorithm.buffer,
            ], f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_algorithm_state(self, folder):
        import pickle
        path = self.algorithm_state_path(folder)
        with open(path, 'r') as f:
            [
                self._algorithm.episode,
                self._algorithm.buffer,
            ] = pickle.load(f)
        self._saved_episode = self._algorithm.episode

    def _save_results_if_need(self, path, ep, eps, sve):
        if (ep > self._saved_episode and (ep + 1) % sve == 0) or (ep == eps):
            self._saved_episode = ep
            self.save(path)

    @staticmethod
    def network_weights_path(folder):
        return os.path.join(folder, NETWORK_WEIGHTS_PATH)

    @staticmethod
    def algorithm_state_path(folder):
        return os.path.join(folder, ALGORITHM_STATE_PATH)
