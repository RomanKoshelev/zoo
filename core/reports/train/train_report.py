import os

import matplotlib.pyplot as plt
import numpy as np

from core.reports.base.base_report import BaseReport
from utils.math_tools import running_mean
from utils.os_tools import make_dir_if_not_exists
from utils.string_tools import hms, thousands, progress


class TrainReport(BaseReport):
    def __init__(self, config, info, history):
        BaseReport.__init__(self, 'train_report', config, info)
        self.history = history

    def make(self):
        i = self.info
        c = self.config

        s = self._header_section()
        s += self._experiment_section(
            [['Start', self._time_date(self.info['start_time'])],
             ['Finish', "%s (%s)" % (self._time_date(self.info['finish_time']), self.info['status'])]])
        s += self._summary_section(
            [['BaseDataset', "%s" % c['ds.train_path']],
             ['Data size', "%s + %s" % (thousands(i['data_size'][0]), thousands(i['data_size'][1]))],
             ['Class num', "%d" % len(c['exp.classes'])],
             ['Image size', c['exp.image_size']],
             ['Train batch', c['train.batch_size']],
             ['Dropout keep', c['nn.dropout.keep_prob']],
             ['Full connect', "%d x %d" % (c['nn.arc.fc1'], c['nn.arc.fc2'])],
             ['', ''],
             ['Train loss/acc', '%.0e / %4.1f%%' % (i['train_loss'], i['train_acc'] * 100)],
             ['Test loss/acc', '%.0e / %4.1f%%' % (i['valid_loss'], i['valid_acc'] * 100)]])
        s += self._section(
            'Progress',
            [['Epoch', "%d / %d" % (i['epoch'] + 1, c['train.epochs'])],
             ['Progress', progress(i['progress'], c['report.width'])],
             ['Time spent', hms(i['spent'])],
             ['      Left', hms(i['left'])],
             ['     Total', hms(i['spent'] + i['left'])],
             ['    Finish', "%s %s" % (i['finish_time'].time(), i['finish_time'].date())],
             ['Processed', "%s imgs" % thousands(i['processed_images'])],
             ['Performance', "%.1f img/sec" % i['performance']],
             ['', "%s per epoch" % hms(i['spent'] / float(i['epoch'] + 1))]])
        s += self._diagram_section()
        s += self._footer_section()
        self.content = s

    def _diagram_section(self):
        s = ""
        s += self._diagram('valid_loss', 4, mean=True, scale='log')
        s += self._diagram('valid_acc', 5, mean=True)
        s += "</br>"
        s += self._diagram('train_loss', 2, mean=True, scale='log')
        s += self._diagram('train_acc', 3, mean=True)
        s += "</br>"
        s += self._diagram('learning_rate', 6, scale='log')
        s += self._diagram('epochs', axe_idx=1, val_idx=0)
        return s

    def _diagram(self, name, val_idx, mean=False, scale='linear', axe_idx=0):
        img = self._create_diagram(name, self.work_path, self.history, axe_idx, val_idx, mean, scale)
        return "<img src='%s' width=600/>\n" % img

    def _create_diagram(self, name, work_dir, arr, x_idx, y_idx, mean=False, scale='linear'):
        path = os.path.join(work_dir, name + ".png")
        x = np.asarray(arr)[:, x_idx] / (60 * 60 if x_idx == 1 else 1)
        y = np.asarray(arr)[:, y_idx] / (60 * 60 if y_idx == 1 else 1)
        plt.clf()
        plt.grid(True)
        plt.yscale(scale)
        plt.title(name)
        plt.xlabel(['epochs', 'time, h'][x_idx])

        if mean:
            m = running_mean(y, self.config['report.mean_frame_epochs'])
            plt.plot(x, y, 'c-', x, m, 'b-')
        else:
            plt.plot(x, y)

        plt.savefig(make_dir_if_not_exists(path))
        return name + ".png"
