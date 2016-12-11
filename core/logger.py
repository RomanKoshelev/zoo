from __future__ import print_function

import os
import pickle

from core.context import Context
from core.reporter import Reporter
from utils.os_tools import make_dir_if_not_exists
from utils.stopwatch import Stopwatch
from utils.string_tools import hms, tab

WORK_DIR = "logger/"


class Logger:
    def __init__(self):
        self._episodes = Context.config['exp.episodes']
        self._work_path = os.path.join(Context.work_path, WORK_DIR)
        self._sw = Stopwatch()
        self._saved_time = 0.
        self._train_history = []
        self._eval_history = []
        self._reporter = Reporter()

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "saved_time: %s" % hms(self._saved_time),
            "train_history: %d" % len(self._train_history),
            "eval_history: %d" % len(self._eval_history),
            "reporter: %s" % tab(self._reporter),
        )

    def on_start(self):
        self._sw.start()

    def on_train_episode(self, e, n, r, q):
        # episode, noise_rate, reward, q_max
        t = self.time_spent
        self._train_history.append([e, t, n, r, q])
        self._log_training_episode(e, n, r, q)
        self._update_training_title(self._episodes, e, n, r, q)

        if (e + 1) % Context.config['report.write_every_episodes'] == 0:
            self._write_html_report()

        if (e + 1) % Context.config['report.summary_every_episodes'] == 0:
            self._log_summary(e, self._episodes, self.time_spent)

        if (e + 1) % Context.config['exp.save_every_episodes'] == 0:
            self._save_state()

    def on_evaluiation_end(self, e, r):
        self._log_evaluiation_end(r)
        self._eval_history.append([e, self.time_spent, r])  # e,t,r

    @property
    def episode(self):
        return len(self._train_history)

    @property
    def time_left(self):
        return int((self.time_spent * (1 - self.progress) / self.progress) if self.progress != 0 else 0)

    @property
    def progress(self):
        return self.episode / float(self._episodes)

    @property
    def time_spent(self):
        return self._saved_time + self._sw.time_elapsed

    @property
    def _state_path(self):
        return os.path.join(self._work_path, "state.pickle")

    def _save_state(self):
        with open(make_dir_if_not_exists(self._state_path), 'w') as f:
            pickle.dump([
                self.time_spent,
                self._train_history,
                self._eval_history
            ], f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self):
        with open(self._state_path, 'r') as f:
            [
                self._saved_time,
                self._train_history,
                self._eval_history
            ] = pickle.load(f)

    def _write_html_report(self):
        info = {
            'ep': len(self._train_history),
            'eps': self._episodes,
            'spent': self.time_spent,
            'left': self.time_left,
            'progress': self.progress,
            'train_history': self._train_history,
            'eval_history': self._eval_history,
        }
        self._reporter.write_html_report(info)

    @staticmethod
    def on_evaluiation_start():
        Context.window_title['episode'] = "| Evaluating ..."

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
        Context.window_title['episode'] = "| %d%% N=%.2f R=%+.0f Q=%+.0f" % (e / float(eps) * 100, n, r, q)

    def _log_training_episode(self, e, n, r, q):
        self.log("Ep: %3d  |  NR: %.2f  |  Reward: %+7.0f  |  Qmax: %+8.1f" % (e, n, r, q))
