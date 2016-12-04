import datetime


def add_secs(tm, secs):
    fulldate = datetime.datetime(tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(seconds=secs)
    return fulldate


def datetime_after_secs(secs):
    return add_secs(datetime.datetime.now(), secs)
