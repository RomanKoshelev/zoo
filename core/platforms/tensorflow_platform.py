import tensorflow as tf

from core.platform import Platform


class TensorflowPlatform(Platform):
    def __init__(self):
        Platform.__init__(self)
        self.session = None

    def __enter__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        return self

    def __exit__(self, *args):
        self.session.close()
