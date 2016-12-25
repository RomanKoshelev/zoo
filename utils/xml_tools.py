from __future__ import print_function

import lxml.html as parser
from lxml.html import tostring


def xml_children_content(xml, xpath):
    try:
        root = parser.fromstring(xml).xpath('//' + xpath)[0]
        return ''.join([tostring(child) for child in root.iterchildren()])
    except IndexError:
        return ""


def xml_content(xml, xpath):
    try:
        root = parser.fromstring(xml).xpath('//' + xpath)[0]
        return tostring(root)
    except IndexError:
        return ""


