import os

from core.context import Context

BASE_PATH = "./out/experiments"


class Experiment:
    def __init__(self, exp_id, proc, ini_from=None):
        self.id = exp_id
        self.ini_id = ini_from if ini_from is not None else exp_id
        self._proc = proc
        self._update_title()

    def start(self):
        self._proc.start(self.init_path, self.work_path)

    def proceed(self):
        self._proc.proceed(self.init_path, self.work_path)

    @property
    def work_path(self):
        return os.path.join(BASE_PATH, self.id)

    @property
    def init_path(self):
        return os.path.join(BASE_PATH, self.ini_id)

    def _update_title(self):
        Context.window_title['exp'] = "|  %s #%s" % (self._proc.__class__.__name__, self.id)
        if self.id != self.ini_id:
            Context.window_title['exp'] += "[%s]" % self.ini_id
