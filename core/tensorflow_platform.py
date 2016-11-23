import tensorflow as tf


class TensorflowPlatform:
    def __init__(self):
        self.session = None

    def __enter__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        return self

    def __exit__(self, *args):
        self.session.close()
