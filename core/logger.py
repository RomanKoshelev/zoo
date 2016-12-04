from __future__ import print_function

from core.context import Context
from utils.string_tools import hms


class Logger:
    def __init__(self):
        pass

    @staticmethod
    def on_evaluiation_start():
        Context.window_title['episode'] = "|  Evaluating ..."

    def _log_evaluiation_end(self, r):
        pass

    def _log_summary(self, ep, eps, spent):
        line = '========================================================================='
        self.log(line + ('\n' * 3))
        self._log_summary_time(ep, eps, spent)
        self.log(('\n' * 3) + line)

    @staticmethod
    def log(text, end='\n'):
        print(text, end=end)

    # noinspection PyStringFormat
    def _log_summary_time(self, ep, eps, spent):
        progress = ep / float(eps)
        left = (spent * (1 - progress) / progress) if progress != 0 else 0
        self.log("Time spent:  %s | " % hms(spent), end='')
        self.log("%.0f%% " % (progress * 100,), end='')
        self.log("+ %s = %s, " % (hms(left), hms(spent + left)), end='')
        self.log("%.1f per sec" % (ep / float(spent),))

    @staticmethod
    def _update_training_title(eps, e, n, r, q):
        Context.window_title['episode'] = "|  %d/%d: N = %.2f, R = %+.0f, Q = %+.0f" % (e, eps, n, r, q)

    def _log_training_episode(self, e, n, r, q):
        self.log("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))
