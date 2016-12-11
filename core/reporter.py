from __future__ import print_function

import os
import pickle
import socket
import types

import matplotlib.pyplot as plt
import numpy as np

from core.context import Context
from core.logger import Logger
from utils.math_tools import running_mean
from utils.os_tools import make_dir_if_not_exists
from utils.stopwatch import Stopwatch
from utils.string_tools import hms
from utils.time_tools import datetime_after_secs

SUMMARY_EVERY_EPISODES = 10
WORK_DIR = "report/"
GROUPS = ['exp', 'alg', 'mind', 'env', 'report']  # 'view'
GROUP_NAMES = {
    'exp': "Experiment",
    'env': "Environment",
    'mind': "Mind",
    'alg': "Algorithm",
    'report': "Reporting",
    'view': "Viewing"
}


class Reporter(Logger):
    def __init__(self, base_path):
        Logger.__init__(self)
        self._episodes = Context.config['exp.episodes']
        self._work_dir = os.path.join(base_path, WORK_DIR)
        self._sw = Stopwatch()
        self._saved_time = 0.
        make_dir_if_not_exists(self._work_dir)
        self._train_history_etnrq = []
        self._eval_history_etr = []
        self.html_path = os.path.abspath(os.path.join(self._work_dir, "report.html"))

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "saved_time: %s" % hms(self._saved_time),
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
            self.write_html_report()

        if (e + 1) % Context.config['report.summary_every_episodes'] == 0:
            self._log_summary(e, self._episodes, self.total_time_elapsed)

        if (e + 1) % Context.config['exp.save_every_episodes'] == 0:
            self._save_state()

    def on_evaluiation_end(self, e, r):
        self._log_evaluiation_end(r)
        self._eval_history_etr.append([e, self.total_time_elapsed, r])

    def write_html_report(self):
        report = "<HTML><HEAD>%s<TITLE>%s</TITLE></HEAD><BODY><H1>%s</H1>\n" % (
            "<meta http-equiv='refresh' content='%d'>" % Context.config['report.refresh_html_every_secs'],
            "Exp #%s" % Context.experiment.id,
            "Experiment #%s" % Context.experiment.id,
        )

        report += self._report_urgent()
        report += self._report_passport()
        report += self._report_config()
        report += self._report_instances()
        report += self._report_progress()
        report += self._report_results()
        report += self._report_diagrams()

        report += "</BODY></HTML>\n"
        with open(self.html_path, 'w') as f:
            f.write(report)

    def _report_progress(self):
        html = "<h2>Progress</h2>\n"
        html += "<pre>\n"

        ep = len(self._train_history_etnrq)
        eps = self._episodes
        spent = self.total_time_elapsed
        progress = ep / float(eps)
        left = int((spent * (1 - progress) / progress) if progress != 0 else 0)

        def field(n, v):
            return "  %-14s %s\n" % (n + ':', v)

        html += field('Episodes', ep)
        html += field('Steps', ep * Context.config['exp.steps'])
        html += field('Total time', hms(spent + left))
        html += field('  spent', "%s (%s)" % (hms(spent), str(int(progress * 100)) + '%'))
        html += field('  left', hms(left))
        html += field('Finish', "%s %s" % (datetime_after_secs(left).time(), datetime_after_secs(left).date()))
        html += field('Performance', "%.2f per sec" % (ep / float(spent),))

        html += "</pre>\n"
        return html

    @property
    def episode(self):
        return len(self._train_history_etnrq)

    @property
    def average_eval_reward(self):
        return self._get_last_mean(self._eval_history_etr, 2)

    @property
    def average_qmax(self):
        return self._get_last_mean(self._train_history_etnrq, 4)

    @property
    def time_left(self):
        return int((self.total_time_elapsed * (1 - self.progress) / self.progress) if self.progress != 0 else 0)

    @property
    def progress(self):
        return self.episode / float(self._episodes)

    def _report_urgent(self):
        def field(n, v):
            return "  %-10s %s\n" % (n + ':', v)
        html = "<font color=red><pre>\n"
        html += field('Reward', '%.0f' % self.average_eval_reward)
        html += field('Qmax', '%.0f' % self.average_qmax)
        html += field('Left', "%s (%.0f%%)" % (hms(self.time_left), self.progress*100))
        html += "</pre></font>\n"
        return html

    def _report_results(self):
        html = "<h2>Results</h2>\n"
        html += "<pre>\n"

        def field(n, v):
            return "  %-14s %s\n" % (n + ':', v)

        html += field('Train reward', '%+12.2f' % self._get_last_mean(self._train_history_etnrq, 3))
        html += field('Eval reward', '%+12.2f' % self._get_last_mean(self._eval_history_etr, 2))

        html += "</pre>\n"
        return html

    @staticmethod
    def _report_passport():
        def field(n, v):
            return "  %-14s %s\n" % (n + ':', v)

        dt = datetime_after_secs(0)
        html = "<font color=#AAAAAA><pre>\n"
        html += field('Report time', dt)
        html += field('Host name', socket.gethostname())
        html += "</pre></font>\n"
        return html

    @staticmethod
    def _report_instances():
        html = "<h2>Instances</h2>\n"
        html += "<pre>\n"
        html += str(Context.experiment).replace('\t', '  ')
        html += "</pre>\n"
        return html

    @staticmethod
    def _report_config():
        def to_str(item):
            if isinstance(item, types.FunctionType):
                return item.__name__
            else:
                return str(item)

        html = "<h2>Configuration</h2>\n"
        html += "<pre>\n"
        for g in GROUPS:
            html += "<b>  %s:</b>\n" % GROUP_NAMES[g]
            for k, v in iter(sorted(Context.config.iteritems())):
                if k.startswith(g + '.'):
                    html += "    %-32s %s\n" % (k + ":", to_str(v))
        html += "</pre>\n"
        return html

    @staticmethod
    def _get_last_mean(arr, idx):
        if len(arr) > 0:
            r = np.asarray(arr)[:, idx]
            l = len(r)
            f = Context.config['report.diagram_mean_frame']
            return np.mean(r[max(0, l - f):l])
        return 0

    def _report_diagrams(self):
        txt = "<h2>Diagrams</h2>\n"
        for d in self._create_all_diagrams():
            txt += "<img src='%s' width=500>\n" % (os.path.basename(d[1]))
        return txt

    def _create_all_diagrams(self, x_idx=0):
        diagrams = []
        if len(self._eval_history_etr) > 0:
            diagrams.append(self._create_mean_diagram('eval_reward', self._eval_history_etr, x_idx, 2))
        if len(self._train_history_etnrq) > 0:
            diagrams.append(self._create_mean_diagram('train_reward', self._train_history_etnrq, x_idx, 3))
            diagrams.append(self._create_diagram('train_noise', self._train_history_etnrq, x_idx, 2))
            diagrams.append(self._create_diagram('train_qmax', self._train_history_etnrq, x_idx, 4))
        return diagrams

    def _create_diagram(self, name, arr, x_idx, y_idx):
        path = os.path.join(self._work_dir, name + ".png")
        x = np.asarray(arr)[:, x_idx]
        y = np.asarray(arr)[:, y_idx]
        plt.clf()
        plt.grid(True)
        plt.plot(x, y)
        plt.title(name)
        plt.xlabel(['episodes', 'time'][x_idx])
        plt.savefig(path)
        return [name, path]

    def _create_mean_diagram(self, name, arr, x_idx, y_idx):
        path = os.path.join(self._work_dir, name + ".png")
        x = np.asarray(arr)[:, x_idx]
        y = np.asarray(arr)[:, y_idx]
        m = running_mean(y, Context.config['report.diagram_mean_frame'])
        plt.clf()
        plt.grid(True)
        plt.plot(x, y, 'c-', x, m, 'b-')
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
