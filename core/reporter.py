from __future__ import print_function

import pickle

from core.context import Context
import os

from utils.os_tools import make_dir_if_not_exists
from utils.stopwatch import Stopwatch, hms

SUMMARY_EVERY_EPISODES = 10
WORK_DIR = "report/"


# todo: save/restore sate: history, last_saved_duration
# todo: draw diagramm for Qmax and Rewards depended of Episodes and Time

class Reporter:
    def __init__(self, base_path):
        self._episodes = Context.config['episodes']
        self._episode = 0
        self._work_dir = os.path.join(base_path, WORK_DIR)
        self._sw = Stopwatch()
        self._saved_time = 0.
        make_dir_if_not_exists(self._work_dir)
        self._train_history = []
        self._eval_history = []

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "saved_time: %02d:%02d:%02d" % hms(self._saved_time),
            "train_history: %d" % len(self._train_history),
            "evaluation_history: %d" % len(self._eval_history),
        )

    def on_start(self):
        self._sw.start()

    def on_train_episode(self, e, n, r, q):
        # episode, noise_rate, reward, q_max
        self._episode = e
        self._train_history.append([e, n, r, q])
        self._log_episode(e, n, r, q)
        self._update_title(e, n, r, q)

        if (e + 1) % Context.config['report.write_every_episodes'] == 0:
            self._write_html_report()

        if (e + 1) % Context.config['report.summary_every_episodes'] == 0:
            self._log_summary()

        if (e + 1) % Context.config['save_every_episodes'] == 0:
            self._save_state()

    def on_evaluiation_start(self):
        self._log("Evaluating...")
        Context.window_title['episode'] = "|  Evaluating ..."

    def on_evaluiation_end(self, e, r):
        self._log("Evaluation done with reward = %.0f" % r)
        Context.window_title['episode'] = "|  Evaluation reward: %.0f" % r
        self._eval_history.append([e, r])

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

    @property
    def total_time_elapsed(self):
        return self._saved_time + self._sw.time_elapsed

    @property
    def _state_path(self):
        return os.path.join(self._work_dir, "state.pickle")

    def _save_state(self):
        with open(self._state_path, 'w') as f:
            pickle.dump([
                self._saved_time,
                self._train_history,
                self._eval_history
            ], f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self):
        try:
            with open(self._state_path, 'r') as f:
                [
                    self._saved_time,
                    self._train_history,
                    self._eval_history
                ] = pickle.load(f)
        except IOError:
            print ("\n\nCan't open file %s\n\n" % self._state_path)

    # noinspection PyStringFormat
    def _log_summary_time(self):
        total = self._episode
        progress = self._episode / float(self._episodes)
        spent = self.total_time_elapsed
        left = (spent * (1 - progress) / progress) if progress != 0 else 0
        spent = self._sw.time_elapsed
        self._log("Time spent:  %02d:%02d:%02d | " % hms(spent), end='')
        self._log("%.0f%% " % (progress * 100,), end='')
        self._log("+ %02d:%02d:%02d = %02d:%02d:%02d, " % (hms(left) + hms(spent + left)), end='')
        self._log("%.1f per sec" % (total / float(spent),))
