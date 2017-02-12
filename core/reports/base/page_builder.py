import os

from asq.initiators import query

from common.utils.os_tools import listdirs

LEFT_MENU_HTML = "left_menu.html"
TOP_MENU_HTML = "top_menu_%s.html"
INDEX_HTML = "index.html"


class PageBuilder:
    def __init__(self, config, report):
        self.config = config
        self._report = report
        self.report_registry = []
        self._register('main_report', 'Main')
        self._register('train_report', 'Train')
        self._register('test_report', 'Test')
        self._register('error_report', 'Errors')
        self._register('signal_report', 'Signals')
        self._left_menu_path = os.path.join(self.config['report.http_home'], LEFT_MENU_HTML)
        self._top_menu_path = os.path.join(self._report.work_path, TOP_MENU_HTML % self._report.exp_id)

    def get_report_name(self, report_id):
        report = query(self.report_registry).first_or_default(None, lambda r: r['id'] == report_id)
        if report is not None:
            return report['name']
        else:
            return None

    def scripts(self):
        return """
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
        <script>
        $(document).ready(function(){
          $("#left_menu").load("%s");
          $("#report_menu").load("%s");
        });
        </script>
        """ % (self._left_menu_path, self._top_menu_path)

    @staticmethod
    def css_styles():
        return """
        <style type='text/css'>
        body, html {padding: 0; margin:0;}
        a.left_menu:link { color:#AAA; text-decoration:none; font-weight:normal; }
        a.left_menu:visited { color: #AAA; text-decoration:none; font-weight:normal; }
        a.left_menu:hover { color: #FFF; text-decoration:underline; font-weight:normal; }
        a.left_menu:active { color: #AAA; text-decoration:none; font-weight:normal; }
        a.top_menu:link { color:#AAA; text-decoration:none; font-weight:normal; }
        a.top_menu:visited { color: #AAA; text-decoration:none; font-weight:normal; }
        a.top_menu:hover { color: #444; text-decoration:underline; font-weight:normal; }
        a.top_menu:active { color: #AAA; text-decoration:none; font-weight:normal; }
        </style>
        """

    @staticmethod
    def start():
        return """
        <table cellpadding=5 width=100% height=100%><tr><td valign=top width=200 bgcolor=#444444>
          <font color=#DDD>
          <pre><div id="left_menu"></div></pre>
          </font>
        </td><td valign=top>
        """

    @staticmethod
    def end():
        return "\n</td></tr></table>"

    def update_files(self):
        self._update_top_menu()
        self._update_left_menu()
        self._update_root()

    def _register(self, rid, name):
        if not query(self.report_registry).any(lambda r: r['id'] == rid):
            self.report_registry.append({
                'id': rid,
                'name': name,
                'path': 'report/%s.html' % rid
            })

    def _update_root(self):
        s = "<html><head>"
        s += "<title>Experiments (%s)</title>" % self.config['report.http_home']
        s += self.scripts()
        s += self.css_styles()
        s += "</head><body>"
        s += self.start()
        s += "<h1>Experiments (%s)</h1>" % self.config['report.http_home']
        s += self.end()
        s += "\n</body></html>"
        with open(os.path.join(self.config['exp.base_path'], INDEX_HTML), 'w') as f:
            f.write(s)

    def _update_left_menu(self):
        exp_group = self.config['exp.group']
        exp_path = self.config['exp.base_path']
        http_home = self.config['report.http_home']
        http_root = self.config['report.http_root']
        s = ""
        s += "<a class=left_menu href='%s'>NetForge</a> | " % http_root
        s += "<a class=left_menu href='%s'>%s experiments</a>" % (http_home, exp_group)
        s += "\n\n"

        for exp_id in listdirs(exp_path):
            s += "  <a class=left_menu href='%s%s/report/main_report.html'>%s</a>\n" % (http_home, exp_id, exp_id)

        with open(self._left_menu_path, 'w') as f:
            f.write(s)

    def _update_top_menu(self):
        def report_link(r):
            if os.path.exists(os.path.join(exp_path, exp_id, r['path'])):
                return "<a class=top_menu href='%s%s/%s'>%s</a>\n" % (http_home, exp_id, r['path'], r['name'])
            else:
                return None

        exp_id = self._report.exp_id
        exp_path = self.config['exp.base_path']
        http_home = self.config['report.http_home']
        s = "<font color=#AAA>&nbsp;"

        menu = []
        for report in self.report_registry:
            l = report_link(report)
            if l is not None:
                menu.append(l)
        s += " | ".join(menu)
        s += "</font>"

        with open(self._top_menu_path, 'w') as f:
            f.write(s)
