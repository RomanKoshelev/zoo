from __future__ import print_function

import os

from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from core.context import Context
from utils.string_tools import tab
from utils.os_tools import provide_dir

MODEL_PATH = "model/weights.ckpt"
TRAIN_STATE_PATH = "train/state.pickle"


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
            self._reporter.on_episode(ep, nr, reward, maxq)
            self._save_results_if_need(work_path, ep,
                                       Context.config['episodes'],
                                       Context.config['mind.save_every_episodes'])

        return self._algorithm.train(Context.config['episodes'], Context.config['steps'], on_episod)

    def save(self, folder):
        self.save_weights(folder)
        self.save_train_state(folder)

    def restore(self, folder):
        self.restore_weights(folder)
        self.restore_train_state(folder)

    def restore_weights(self, folder):
        self._algorithm.restore_weights(self.weights_path(folder))

    def save_weights(self, folder):
        self._algorithm.save_weights(provide_dir(self.weights_path(folder)))

    def save_train_state(self, folder):
        import pickle
        path = self.train_state_path(folder)
        provide_dir(path)
        with open(path, 'w') as f:
            pickle.dump([
                self._algorithm.episode,
                self._algorithm.buffer,
            ], f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_train_state(self, folder):
        import pickle
        path = self.train_state_path(folder)
        with open(path, 'r') as f:
            [
                self._algorithm.episode,
                self._algorithm.buffer,
            ] = pickle.load(f)
        self._saved_episode = self._algorithm.episode

    def _save_results_if_need(self, path, ep, eps, sve):
        if (ep > self._saved_episode and (ep + 1) % sve == 0) or (ep == eps):
            self._reporter.on_save_start(path)
            self._saved_episode = ep
            self.save(path)
            self._reporter.on_save_done(path)

    @staticmethod
    def weights_path(folder):
        return os.path.join(folder, MODEL_PATH)

    @staticmethod
    def train_state_path(folder):
        return os.path.join(folder, TRAIN_STATE_PATH)
