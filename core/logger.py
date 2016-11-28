from __future__ import print_function

from core.context import Context
from utils.stopwatch import hms


class Logger:
    def __init__(self):
        pass

    def on_evaluiation_start(self):
        self._log("Evaluating...")
        Context.window_title['episode'] = "|  Evaluating ..."

    def _log_summary(self, ep, eps, spent):
        line = '========================================================================='
        self._log(line + ('\n' * 3))
        self._log_summary_time(ep, eps, spent)
        self._log(('\n' * 3) + line)

    @staticmethod
    def _log(text, end='\n'):
        print(text, end=end)

    # noinspection PyStringFormat
    def _log_summary_time(self, ep, eps, spent):
        progress = ep / float(eps)
        left = (spent * (1 - progress) / progress) if progress != 0 else 0
        self._log("Time spent:  %02d:%02d:%02d | " % hms(spent), end='')
        self._log("%.0f%% " % (progress * 100,), end='')
        self._log("+ %02d:%02d:%02d = %02d:%02d:%02d, " % (hms(left) + hms(spent + left)), end='')
        self._log("%.1f per sec" % (ep / float(spent),))

    def on_save_start(self, what, path):
        self._log("Saving %s to '%s' ..." % (what, path))

    def on_save_done(self, what, path):
        self._log("Saved %s to '%s'" % (what, path))

    def on_restore_done(self, what, path):
        self._log("Restored %s from '%s'" % (what, path))

    @staticmethod
    def _update_training_title(eps, e, n, r, q):
        Context.window_title['episode'] = "|  %d/%d: N = %.2f, R = %+.0f, Q = %+.0f" % (e, eps, n, r, q)

    def _log_training_episode(self, e, n, r, q):
        self._log("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))

    def _log_evaluiation_end(self, r):
        self._log("Evaluation done with reward = %.0f" % r)
        Context.window_title['episode'] = "|  Evaluation reward: %.0f" % r
