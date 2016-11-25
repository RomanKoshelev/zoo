import tensorflow as tf


class TensorflowPlatform:
    def __init__(self):
        self.session = None

    def __str__(self):
        return "%s" % (
            self.__class__.__name__,
        )

    def __enter__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        return self

    # noinspection PyUnusedLocal
    def __exit__(self, *args):
        self.session.close()
