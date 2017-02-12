from __future__ import print_function

from core.reports.base.base_report import BaseReport


class MainReport(BaseReport):
    def __init__(self, config):
        BaseReport.__init__(self, 'main_report', config, {})

    def make(self):
        s = self._header_section()
        s += self._experiment_section()
        s += self._config_section()
        s += self._footer_section()
        self.content = s
