from __future__ import print_function
from core.context import Context
import os

from utils.os_tools import make_dir_if_not_exists
from utils.stopwatch import Stopwatch, hms

SUMMARY_EVERY_EPISODES = 10
WORK_DIR = "report/"


class Reporter:
    def __init__(self, base_path):
        self._episodes = Context.config['episodes']
        self._episode = 0
        self._work_dir = os.path.join(base_path, WORK_DIR)
        self._sw = Stopwatch()
        make_dir_if_not_exists(self._work_dir)
        self._history = []

    def __str__(self):
        return self.__class__.__name__

    def on_start(self):
        self._sw.start()

    def on_episode(self, e, n, r, q):
        self._episode = e
        self._history.append([e, n, r, q])
        self._log_episode(e, n, r, q)
        self._update_title(e, n, r, q)

        if (e + 1) % Context.config['report.write_every_episodes'] == 0:
            self._write_html_report()

        if (e + 1) % Context.config['report.summary_every_episodes'] == 0:
            self._log_summary()

    def on_save_start(self, what, path):
        self._log("Saving %s to '%s' ..." % (what, path))

    def on_save_done(self, what, path):
        self._log("Saved %s to '%s'" % (what, path))

    def _update_title(self, e, n, r, q):
        Context.window_title['episode'] = "|  %d/%d: N = %.2f, R = %+.0f, Q = %+.0f" % (e, self._episodes, n, r, q)

    def _log_episode(self, e, n, r, q):
        self._log("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))

    # noinspection PyMethodMayBeStatic
    def _log(self, text, end='\n'):
        print(text, end=end)

    def _write_html_report(self):
        report = "<HTML><BODY><H1>REPORT</H1></BODY></HTML>"
        html_path = os.path.join(self._work_dir, "report.html")
        with open(html_path, 'w') as f:
            f.write(report)
        self._log("%s is wrote '%s'" % (self.__class__.__name__, html_path))

    def _log_summary(self):
        line = '========================================================================='
        self._log(line + ('\n' * 3))

        self._log_summary_time()

        self._log(('\n' * 3) + line)

    # noinspection PyStringFormat
    def _log_summary_time(self):
        total = self._episode
        progress = self._episode / float(self._episodes)
        spent = self._sw.time_elapsed
        left = (spent * (1 - progress) / progress) if progress != 0 else 0
        spent = self._sw.time_elapsed
        self._log("Time spent:  %02d:%02d:%02d | " % hms(spent), end='')
        self._log("%.0f%% " % (progress * 100,), end='')
        self._log("+ %02d:%02d:%02d = %02d:%02d:%02d, " % (hms(left) + hms(spent + left)), end='')
        self._log("%.1f per sec" % (total / float(spent),))
