import socket
import types
import os

import matplotlib.pyplot as plt
import numpy as np

from core.context import Context
from utils.math_tools import running_mean
from utils.os_tools import make_dir_if_not_exists
from utils.string_tools import hms
from utils.time_tools import datetime_after_secs

GROUPS = ['exp', 'alg', 'mind', 'env', 'report']  # 'view'
GROUP_NAMES = {
    'exp': "Experiment",
    'env': "Environment",
    'mind': "Mind",
    'alg': "Algorithm",
    'report': "Reporting",
    'view': "Viewing"
}

WORK_DIR = "reporter/"
REPORT_FILE = "report.html"


class Reporter:
    def __init__(self):
        self._work_path = os.path.join(Context.work_path, WORK_DIR)
        self.html_path = os.path.abspath(os.path.join(self._work_path, REPORT_FILE))

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "html_path: %s" % self.html_path,
        )

    def write_html_report(self, info):
        s = "<HTML><HEAD>%s<TITLE>%s</TITLE></HEAD><BODY><H1>%s</H1>\n" % (
            "<meta http-equiv='refresh' content='%d'>" % Context.config['report.refresh_html_every_secs'],
            "Exp #%s" % Context.experiment.id,
            "Experiment #%s" % Context.experiment.id,
        )

        s += self._urgent_section(info)
        s += self._passport_section()
        s += self._config_section()
        s += self._progress_section(info)
        s += self._results_section(info)
        s += self._diagrams_section(info)
        s += self._instances_section()

        s += "</BODY></HTML>\n"
        with open(make_dir_if_not_exists(self.html_path), 'w') as f:
            f.write(s)

    @staticmethod
    def field(n, v):
        return "  %-14s %s\n" % (n + (':' if len(n) > 0 else ''), v)

    def _progress_section(self, info):
        ep = info['ep']
        eps = info['eps']
        spent = info['spent']
        progress = ep / float(eps)
        left = int((spent * (1 - progress) / progress) if progress != 0 else 0)
        s = "<h2>Progress</h2>\n"
        s += "<pre>\n"
        s += self.field('Episodes', ep)
        s += self.field('Steps', ep * Context.config['exp.steps'])
        s += self.field('Total time', hms(spent + left))
        s += self.field('  spent', "%s (%s)" % (hms(spent), str(int(progress * 100)) + '%'))
        s += self.field('  left', hms(left))
        s += self.field('Finish', "%s %s" % (datetime_after_secs(left).time(), datetime_after_secs(left).date()))
        s += self.field('Performance', "%.2f per sec" % (ep / float(spent),))
        s += "</pre>\n"
        return s

    def _urgent_section(self, info):
        s = "<font color=red><pre>\n"
        s += self.field('Reward', '%.0f' % self._get_last_mean(info['eval_history'], 2))
        s += self.field('Qmax', '%.0f' % self._get_last_mean(info['train_history'], 4))
        s += self.field('Left', "%s (%.0f%%)" % (hms(info['left']), info['progress'] * 100))
        s += "</pre></font>\n"
        return s

    def _results_section(self, info):
        s = "<h2>Results</h2>\n"
        s += "<pre>\n"
        s += self.field('Train reward', '%+12.2f' % self._get_last_mean(info['train_history'], 3))
        s += self.field('Eval reward', '%+12.2f' % self._get_last_mean(info['eval_history'], 2))
        s += "</pre>\n"
        return s

    def _passport_section(self):
        dt = datetime_after_secs(0)
        s = "<font color=#AAAAAA><pre>\n"
        s += self.field('Report time', dt)
        s += self.field('Host name', socket.gethostname())
        s += "</pre></font>\n"
        return s

    @staticmethod
    def _instances_section():
        html = "<h2>Instances</h2>\n"
        html += "<pre>\n"
        html += str(Context.experiment).replace('\t', '  ')
        html += "</pre>\n"
        return html

    @staticmethod
    def _config_section():
        def to_str(item):
            if isinstance(item, types.FunctionType):
                return item.__name__
            else:
                return str(item)

        s = "<h2>Configuration</h2>\n"
        s += "<pre>\n"
        for g in GROUPS:
            s += "<b>  %s:</b>\n" % GROUP_NAMES[g]
            for k, v in iter(sorted(Context.config.items())):
                if k.startswith(g + '.'):
                    s += "    %-32s %s\n" % (k + ":", to_str(v))
        s += "</pre>\n"
        return s

    @staticmethod
    def _get_last_mean(arr, idx):
        if len(arr) > 0:
            r = np.asarray(arr)[:, idx]
            l = len(r)
            f = Context.config['report.diagram_mean_frame']
            return np.mean(r[max(0, l - f):l])
        return 0

    def _diagrams_section(self, info):
        txt = "<h2>Diagrams</h2>\n"
        for d in self._create_all_diagrams(info):
            txt += "<img src='%s' width=500>\n" % (os.path.basename(d[1]))
        return txt

    def _create_all_diagrams(self, info, x_idx=0):
        ds = []
        th = info['train_history']
        eh = info['eval_history']
        if len(eh) > 0:
            ds.append(self._create_mean_diagram('eval_reward', eh, x_idx, 2))
        if len(th) > 0:
            ds.append(self._create_mean_diagram('train_reward', th, x_idx, 3))
            ds.append(self._create_diagram('train_noise', th, x_idx, 2))
            ds.append(self._create_diagram('train_qmax', th, x_idx, 4))
        return ds

    def _create_diagram(self, name, arr, x_idx, y_idx):
        path = os.path.join(self._work_path, name + ".png")
        x = np.asarray(arr)[:, x_idx]
        y = np.asarray(arr)[:, y_idx]
        plt.clf()
        plt.grid(True)
        plt.plot(x, y)
        plt.title(name)
        plt.xlabel(['episodes', 'time'][x_idx])
        plt.savefig(make_dir_if_not_exists(path))
        return [name, path]

    def _create_mean_diagram(self, name, arr, x_idx, y_idx):
        path = os.path.join(self._work_path, name + ".png")
        x = np.asarray(arr)[:, x_idx]
        y = np.asarray(arr)[:, y_idx]
        m = running_mean(y, Context.config['report.diagram_mean_frame'])
        plt.clf()
        plt.grid(True)
        plt.plot(x, y, 'c-', x, m, 'b-')
        plt.title(name)
        plt.xlabel(['episodes', 'time'][x_idx])
        plt.savefig(make_dir_if_not_exists(path))
        return [name, path]
