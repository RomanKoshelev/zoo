import os

from core.context import Context


class Experiment:
    def __init__(self, exp_id, proc, init_from=None):
        self.id = exp_id
        self.init_id = init_from if init_from is not None else exp_id
        self._proc = proc
        self._base_path = Context.config['exp.base_path']
        Context.work_path = self.work_path
        self._update_title()

    def start(self):
        self._proc.start(self.init_path, self.work_path)

    def proceed(self):
        self._proc.proceed(self.init_path, self.work_path)

    @property
    def work_path(self):
        return os.path.join(self._base_path, self.id)

    @property
    def init_path(self):
        return os.path.join(self._base_path, self.init_id)

    def _update_title(self):
        Context.window_title['exp'] = "|  %s #%s" % (self._proc.__class__.__name__, self.id)
        if self.id != self.init_id:
            Context.window_title['exp'] += "[%s]" % self.init_id
