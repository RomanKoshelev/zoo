def tab(obj):
    return str(obj).replace('\n\t', '\n\t\t')


def hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def thousands(i):
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    return locale.format("%d", i, grouping=True)


def progress(p, n):
    return "[%s %.0f%% %s]" % ('.' * int(p * n / 2), p * 100, '.' * int((1 - p) * n / 2))
