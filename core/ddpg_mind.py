from __future__ import print_function
from alg.ddpg_peter_kovacs.ddpg import DDPG_PeterKovacs
from core.context import Context


class DdpgMind:
    def __init__(self, platform, world):
        self.platform = platform
        self.world = world
        self._algorithm = None

    def __enter__(self):
        self._algorithm = DDPG_PeterKovacs(self.platform.session, self.world)
        return self

    def __exit__(self, *args):
        pass

    def train(self, weigts_path):
        def log_episode(e, n, r, q):
            print("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))

        def update_title(e, n, r, q):
            eps = Context.config['episodes']
            Context.window_title['episod'] = "|  %d/%d: R = %+.0f, N = %.2f, Q = %+.0f" % (e, eps, n, r, q)

        def save_results(ep):
            eps = Context.config['episodes']
            sve = Context.config['train.save_every_episodes']
            if (ep > 0 and ep % sve == 0) or (ep == eps - 1):
                self.save(weigts_path)

        def on_episod(ep, reward, nr, maxq):
            log_episode(ep, nr, reward, maxq)
            update_title(ep, nr, reward, maxq)
            save_results(ep)

        episodes = Context.config['episodes']
        steps = Context.config['steps']
        return self._algorithm.train(episodes, steps, on_episod)

    def save(self, folder):
        print("\nSaving [%s].." % folder)
        self.save_weights(folder)
        print("Done.\n")

    def restore(self, folder):
        print("\nRestoring [%s].." % folder)
        self.restore_weights(folder)
        print("Done.\n")

    def restore_weights(self, folder):
        self._algorithm.restore(self.weights_path(folder))

    def save_weights(self, folder):
        self._algorithm.save(self.weights_path(folder))

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self.world.scale_action(a)

    @staticmethod
    def weights_path(folder):
        import os
        return os.path.join(folder, "weights.ckpt")
