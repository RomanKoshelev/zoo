import tensorflow as tf

from core.context import Context


class TensorflowPlatform:
    def __init__(self):
        Context.platform = self
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def __str__(self):
        return "%s" % (
            self.__class__.__name__,
        )

    def __del__(self):
        self.session.close()
