from __future__ import print_function
from core.context import Context


class Reporter:
    def __init__(self):
        pass

    def __str__(self):
        return "%s" % (
            self.__class__.__name__,
        )

    @staticmethod
    def on_episode(e, n, r, q):
        print("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))
        eps = Context.config['episodes']
        Context.window_title['episode'] = "|  %d/%d: N = %.2f, R = %+.0f, Q = %+.0f" % (e, eps, n, r, q)

    @staticmethod
    def on_save_start(path):
        print("Saving [%s] ..." % path)

    @staticmethod
    def on_save_done(path):
        print("Saved [%s]" % path)
