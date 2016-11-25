import os

from core.context import Context

BASE_PATH = "./out/experiments"


class Experiment:
    def __init__(self, exp_id, proc, ini_from=None):
        self.proc = proc
        self.id = exp_id
        self.ini_id = ini_from if ini_from is not None else exp_id
        self._make_folder(self.work_path)
        self._update_title()

    def start(self):
        self.proc.start(self.work_path)

    def proceed(self):
        self.proc.proceed(self.init_path, self.work_path)

    @property
    def work_path(self):
        return os.path.join(BASE_PATH, self.id)

    @property
    def init_path(self):
        return os.path.join(BASE_PATH, self.ini_id)

    @staticmethod
    def _make_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _update_title(self):
        Context.window_title['exp'] = "|  %s #%s" % (self.proc.__class__.__name__, self.id)
        if self.id != self.ini_id:
            Context.window_title['exp'] += "[%s]" % self.ini_id
