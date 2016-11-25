from __future__ import print_function

import os

from alg.ddpg_peter_kovacs.ddpg import DDPG_PeterKovacs
from core.context import Context
from utils.os_tools import provide_folder

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

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self.world.scale_action(a)

    def train(self, work_path):
        def log_episode(e, n, r, q):
            print("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))

        def update_title(e, n, r, q):
            eps = Context.config['episodes']
            Context.window_title['episod'] = "|  %d/%d: R = %+.0f, N = %.2f, Q = %+.0f" % (e, eps, n, r, q)

        def save_results(ep):
            eps = Context.config['episodes']
            sve = Context.config['train.save_every_episodes']
            if (ep > self._saved_episode and ep % sve == 0) or (ep == eps - 1):
                self._saved_episode = ep
                self.save(work_path)

        def on_episod(ep, reward, nr, maxq):
            log_episode(ep, nr, reward, maxq)
            update_title(ep, nr, reward, maxq)
            save_results(ep)

        first_episide = self._saved_episode + 1 if self._saved_episode is not None else 0
        last_episode = Context.config['episodes']
        steps = Context.config['steps']
        return self._algorithm.train(first_episide, last_episode, steps, on_episod)

    def save(self, folder):
        print("\nSaving [%s].." % folder)
        self.save_weights(folder)
        self.save_train_state(folder)
        print("Done.\n")

    def restore(self, folder):
        print("\nRestoring [%s].." % folder)
        self.restore_weights(folder)
        self.restore_train_state(folder)
        print("Done.\n")

    def restore_weights(self, folder):
        self._algorithm.restore(self.weights_path(folder))

    def save_weights(self, folder):
        path = self.weights_path(folder)
        provide_folder(path)
        self._algorithm.save(path)

    def save_train_state(self, folder):
        import pickle
        path = self.train_state_path(folder)
        provide_folder(path)
        with open(path, 'w') as f:
            pickle.dump([self._saved_episode], f)

    def restore_train_state(self, folder):
        import pickle
        path = self.train_state_path(folder)
        with open(path, 'r') as f:
            [self._saved_episode] = pickle.load(f)

    @staticmethod
    def weights_path(folder):
        return os.path.join(folder, MODEL_PATH)

    @staticmethod
    def train_state_path(folder):
        return os.path.join(folder, TRAIN_STATE_PATH)
