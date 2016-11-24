import os

from core.context import Context

BASE_PATH = "./out/experiments"


class Experiment:
    def __init__(self, exp_id, proc, ini_from=None):
        self.proc = proc
        self.id = exp_id
        self.ini_id = ini_from if ini_from is not None else exp_id
        self._make_folder(self.out_path)
        self._update_title()

    def __call__(self):
        self.proc(self.ini_path, self.out_path)

    def _update_title(self):
        Context.window_title['exp'] = "|  %s #%s" % (self.proc.__class__.__name__, self.id)

    @property
    def out_path(self):
        return os.path.join(BASE_PATH, self.id)

    @property
    def ini_path(self):
        return os.path.join(BASE_PATH, self.ini_id)

    @property
    def setting(self):
        return self.proc.__class__.__name__ + "_" + self.proc.setting

    @staticmethod
    def _make_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
