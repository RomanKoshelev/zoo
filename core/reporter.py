from __future__ import print_function

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from core.context import Context
from core.logger import Logger
from utils.os_tools import make_dir_if_not_exists
from utils.stopwatch import Stopwatch, hms

SUMMARY_EVERY_EPISODES = 10
WORK_DIR = "report/"


class Reporter(Logger):
    def __init__(self, base_path):
        Logger.__init__(self)
        self._episodes = Context.config['episodes']
        self._work_dir = os.path.join(base_path, WORK_DIR)
        self._sw = Stopwatch()
        self._saved_time = 0.
        make_dir_if_not_exists(self._work_dir)
        self._train_history_etnrq = []
        self._eval_history_etr = []

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "saved_time: %02d:%02d:%02d" % hms(self._saved_time),
            "train_history: %d" % len(self._train_history_etnrq),
            "evaluation_history: %d" % len(self._eval_history_etr),
        )

    def on_start(self):
        self._sw.start()

    def on_train_episode(self, e, n, r, q):
        # episode, noise_rate, reward, q_max
        t = self.total_time_elapsed
        self._train_history_etnrq.append([e, t, n, r, q])
        self._log_training_episode(e, n, r, q)
        self._update_training_title(self._episodes, e, n, r, q)

        if (e + 1) % Context.config['report.write_every_episodes'] == 0:
            self._write_html_report()

        if (e + 1) % Context.config['report.summary_every_episodes'] == 0:
            self._log_summary(e, self._episodes, self.total_time_elapsed)

        if (e + 1) % Context.config['save_every_episodes'] == 0:
            self._save_state()

    def on_evaluiation_end(self, e, r):
        self._log_evaluiation_end(r)
        self._eval_history_etr.append([e, self.total_time_elapsed, r])

    def _write_html_report(self):
        title = "Experiment %s" % Context.experiment.id
        report = "<HTML><HEAD><TITLE>%s</TITLE></HEAD><BODY><H1>%s</H1>\n" % (title, title)

        report += self._diagrams()

        report += "</BODY></HTML>\n"
        html_path = os.path.join(self._work_dir, "report.html")
        with open(html_path, 'w') as f:
            f.write(report)

    def _diagrams(self):
        txt = "<h2>Diagrams</h2>\n"
        for d in self._create_all_diagrams():
            txt += "<img src='%s' width=500>\n" % (os.path.basename(d[1]))
        return txt

    def _create_all_diagrams(self, x_idx=0):
        diagrams = []
        if len(self._eval_history_etr) > 0:
            diagrams.append(self._create_diagram('eval_reward', self._eval_history_etr, x_idx, 2))
        if len(self._train_history_etnrq) > 0:
            diagrams.append(self._create_diagram('train_reward', self._train_history_etnrq, x_idx, 3))
            diagrams.append(self._create_diagram('train_noise', self._train_history_etnrq, x_idx, 2))
            diagrams.append(self._create_diagram('train_qmax', self._train_history_etnrq, x_idx, 4))
        return diagrams

    def _create_diagram(self, name, arr, x_idx, y_idx):
        path = os.path.join(self._work_dir, name + ".png")
        x = np.asarray(arr)[:, x_idx]
        y = np.asarray(arr)[:, y_idx]
        plt.clf()
        plt.plot(x, y)
        plt.title(name)
        plt.xlabel(['episodes', 'time'][x_idx])
        plt.savefig(path)
        return [name, path]

    @property
    def total_time_elapsed(self):
        return self._saved_time + self._sw.time_elapsed

    @property
    def _state_path(self):
        return os.path.join(self._work_dir, "state.pickle")

    def _save_state(self):
        with open(self._state_path, 'w') as f:
            pickle.dump([
                self.total_time_elapsed,
                self._train_history_etnrq,
                self._eval_history_etr
            ], f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self):
        with open(self._state_path, 'r') as f:
            [
                self._saved_time,
                self._train_history_etnrq,
                self._eval_history_etr
            ] = pickle.load(f)
