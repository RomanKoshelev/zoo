import datetime
import inspect
import os
import socket

from core.reports.base.page_builder import PageBuilder
from utils.lazyproperty import lazyproperty
from utils.os_tools import make_dir_if_not_exists
from utils.time_tools import datetime_after_secs

GROUPS = ['exp', 'ds', 'nn', 'train', 'test', 'report', 'log']
GROUP_NAMES = {
    'exp': "Experiment",
    'ds': "BaseDataset",
    'nn': "BaseNetwork",
    'report': "Reporting",
    'train': "Training",
    'test': "Testing",
    'log': "Logging",
}


class BaseReport:
    def __init__(self, report_id, config: dict, info: dict, dataset_path=None):
        self.config = config
        self.exp_id = config['exp.id']
        self.report_id = report_id
        self.work_path = os.path.join(config['exp.base_path'], config['exp.id'], 'reporter')
        self.info = info
        self.content = None
        self.dataset_path = dataset_path

    @lazyproperty
    def page(self) -> PageBuilder:
        return PageBuilder(self.config, self)

    @property
    def name(self):
        return self.page.get_report_name(self.report_id)

    def _header_section(self):
        s = "<html><head><title>#%s-%s</title>" % (self.exp_id, self.report_id)
        s += self.page.scripts()
        s += self.page.css_styles()
        s += "</head><body bgcolor=#FFFFFF><font face='courier'>"
        s += self.page.start()
        s += self._header()
        return s

    def _header(self):
        s = ""
        s += "<table width=100%%><tr><td bgcolor='#888'>" \
             "&nbsp;<font size=+2 color=#FFF>%s</font>" \
             "</td></tr></table>" \
             % self.exp_id
        s += "<table width=100%%><tr><td bgcolor='#CCC'>" \
             "&nbsp;<font size=+4 color=#FFF>%s</font>" \
             "</td></tr></table>" \
             % self.name
        s += "<div id='report_menu'></div>"
        return s

    @staticmethod
    def _field(n, v):
        return "  %-16s %s\n" % (n + (':' if len(n) > 0 else ''), v)

    def _section(self, title, records, color="#000000"):
        s = ""
        if title is not None:
            s += "<h2>%s</h2>\n" % title
        s += "<font color=%s><pre>\n" % color
        for rec in records:
            s += self._field(rec[0], rec[1])
        s += "</pre></font>\n"
        return s

    def _experiment_section(self, new=None):
        r = [['Description', self.config['exp.description']],
             ['Host name', socket.gethostname()]]
        if new is not None:
            for n in new:
                r.append(n)
        r.append(['Reported at', self._time_date(datetime_after_secs(0), show_date=True)])
        return self._section(None, r, '#888888')

    def _summary_section(self, r):
        return self._section(title='Summary', records=r, color='red')

    def _config_section(self):
        def to_str(o):
            if inspect.isclass(o) or inspect.ismethod(o) or inspect.isfunction(o):
                return "<i>%s</i>.%s" % (o.__module__, o.__name__)
            else:
                return str(o)

        s = "<h2>Configuration</h2>\n"
        s += "<pre>\n"
        for g in GROUPS:
            s += "<b>  %s:</b>\n" % GROUP_NAMES[g]
            for k, v in iter(sorted(self.config.items())):
                if k.startswith(g + '.'):
                    s += "    %-32s %s\n" % (k + ":", to_str(v)[:self.config['report.width']])
        s += "</pre>\n"
        return s

    @staticmethod
    def _time_date(dt, show_date=False):
        if show_date or dt.date() != datetime.datetime.now().date():
            return "%s %s" % (dt.time(), dt.date())
        return "%s" % dt.time()

    def _footer_section(self):
        return self._footer() + self.page.end() + "</font></body></html>"

    @staticmethod
    def _footer():
        return "<br/><br/><br/><table width=100%%><tr><td bgcolor='#CCC'>" \
               "<font color=#FFF>&nbsp;</font></td></tr></table>"

    def make(self):
        raise NotImplementedError

    def save(self):
        with open(make_dir_if_not_exists(self._report_path), "w") as f:
            f.write(self.content)
        self.page.update_files()

    def _make_classified_img_path(self, img_path):
        class_image = '/'.join(img_path.split('/')[-2:])
        return os.path.join(self.dataset_path, class_image)

    @property
    def _report_path(self):
        return os.path.join(self.work_path, "%s.html" % self.report_id)
