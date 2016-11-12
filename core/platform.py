class Platform(object):
    def __init__(self):
        pass

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *args):
        raise NotImplementedError

