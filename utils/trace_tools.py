import traceback


def _make_names(func_name_len, *expr):
    (filename, line_number, function_name, text) = traceback.extract_stack()[-3]
    begin = func_name_len + 1
    end = text.find(')', begin)
    text = [name.strip() for name in text[begin:end].split(',')]
    return zip(text, expr)


def trace_var(var):
    for n in _make_names(len(trace_var.__name__), var):
        print("%s: %s" % (n[0], n[1]))
