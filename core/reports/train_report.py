import os

import matplotlib.pyplot as plt
import numpy as np

from core.reports.base.base_report import BaseReport
from utils.math_tools import running_mean
from utils.os_tools import make_dir_if_not_exists
from utils.string_tools import hms, thousands, progress


class TrainReport(BaseReport):
    def __init__(self, config, info):
        BaseReport.__init__(self, 'train_report', config, info)

    def make(self):
        i = self.info
        c = self.config

        s = self._header_section()
        s += self._experiment_section()
        s += self._section(
            'Progress',
            [['Epoch', "%d / %d" % (i['ep'] + 1, i['eps'])],
             ['Progress', progress(i['progress'], c['report.width'])],
             ['Time spent', hms(i['spent'])],
             ['      Left', hms(i['left'])],
             ['     Total', hms(i['spent'] + i['left'])]
             ])
        s += self._diagram_section()
        s += self._footer_section()
        self.content = s

    def _diagram_section(self, info):
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

    # def _diagram_section(self):
    #     s = ""
    #     s += self._diagram('valid_loss', 4, mean=True, scale='log')
    #     s += self._diagram('valid_acc', 5, mean=True)
    #     s += "</br>"
    #     s += self._diagram('train_loss', 2, mean=True, scale='log')
    #     s += self._diagram('train_acc', 3, mean=True)
    #     s += "</br>"
    #     s += self._diagram('learning_rate', 6, scale='log')
    #     s += self._diagram('epochs', axe_idx=1, val_idx=0)
    #     return s
    #
    # def _diagram(self, name, val_idx, mean=False, scale='linear', axe_idx=0):
    #     img = self._create_diagram(name, self.work_path, self.history, axe_idx, val_idx, mean, scale)
    #     return "<img src='%s' width=600/>\n" % img
    #
    # def _create_diagram(self, name, work_dir, arr, x_idx, y_idx, mean=False, scale='linear'):
    #     path = os.path.join(work_dir, name + ".png")
    #     x = np.asarray(arr)[:, x_idx] / (60 * 60 if x_idx == 1 else 1)
    #     y = np.asarray(arr)[:, y_idx] / (60 * 60 if y_idx == 1 else 1)
    #     plt.clf()
    #     plt.grid(True)
    #     plt.yscale(scale)
    #     plt.title(name)
    #     plt.xlabel(['epochs', 'time, h'][x_idx])
    #
    #     if mean:
    #         m = running_mean(y, self.config['report.mean_frame_epochs'])
    #         plt.plot(x, y, 'c-', x, m, 'b-')
    #     else:
    #         plt.plot(x, y)
    #
    #     plt.savefig(make_dir_if_not_exists(path))
    #     return name + ".png"
