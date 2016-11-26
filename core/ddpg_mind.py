from __future__ import print_function

import os

from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from core.context import Context
from utils.string_tools import tab
from utils.os_tools import provide_dir

MODEL_PATH = "model/weights.ckpt"
TRAIN_STATE_PATH = "state/train.pickle"


class DdpgMind:
    def __init__(self, platform, world):
        self.platform = platform
        self.world = world
        self._algorithm = None
        self._saved_episode = None

    def __enter__(self):
        self._algorithm = DDPG_PeterKovacs(self.platform.session, self.world)
        return self

    def __exit__(self, *args):
        pass

    def __str__(self):
        return "%s\n\t%s\n\t%s\n\t%s\n" % (
            self.__class__.__name__,
            "platform: " + tab(self.platform),
            "world: " + tab(self.world),
            "algorithm: " + tab(self._algorithm),
        )

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self.world.scale_action(a)

    def train(self, work_path):
        def log_episode(e, n, r, q):
            print("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))

        def update_title(e, n, r, q):
            eps = Context.config['episodes']
            Context.window_title['episod'] = "|  %d/%d: N = %.2f, R = %+.0f, Q = %+.0f" % (e, eps, n, r, q)

        def save_results(ep):
            eps = Context.config['episodes']
            sve = Context.config['mind.save_every_episodes']
            if (ep > self._saved_episode and (ep + 1) % sve == 0) or (ep == eps):
                print("\nSaving [%s] ...\n" % work_path)
                self._saved_episode = ep
                self.save(work_path)

        def on_episod(ep, reward, nr, maxq):
            log_episode(ep, nr, reward, maxq)
            save_results(ep)
            update_title(ep, nr, reward, maxq)

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

    @staticmethod
    def weights_path(folder):
        return os.path.join(folder, MODEL_PATH)

    @staticmethod
    def train_state_path(folder):
        return os.path.join(folder, TRAIN_STATE_PATH)
