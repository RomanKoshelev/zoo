import tensorflow as tf


class TensorflowPlatform(object):
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

    def __exit__(self):
        self.session.close()
