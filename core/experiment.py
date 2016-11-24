import os

from core.context import Context

BASE_PATH = "./out/experiments"


class Experiment:
    def __init__(self, name, proc, descr=""):
        self.proc = proc
        self.name = name
        self.descr = descr
        self._make_folder(self.path)

    def __call__(self):
        Context.window_title['exp'] = "|  %s #%s" % (self.proc.__class__.__name__, self.name)
        self.proc(path=self.path)

    @property
    def path(self):
        return os.path.join(BASE_PATH, self.name, self.setting)

    @property
    def setting(self):
        return self.proc.__class__.__name__ + "_" + self.proc.setting

    @staticmethod
    def _make_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
