import lxml.html as parser
from lxml.html import tostring


def xml_children_content(xml, xpath):
    try:
        root = parser.fromstring(xml).xpath('//' + xpath)[0]
        return ''.join([str(tostring(child)) for child in root.iterchildren()])
    except IndexError:
        return ""


def xml_content(xml, xpath):
    try:
        root = parser.fromstring(xml).xpath('//' + xpath)[0]
        return str(tostring(root))
    except IndexError:
        return ""

