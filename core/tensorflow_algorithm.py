import tensorflow as tf
import os


class TensorflowAlgorithm(object):
    def __init__(self, sess, suffix):
        self.sess = sess
        self.suffix = suffix

    def __str__(self):
        return "%s" % (
            self.__class__.__name__,
        )

    @property
    def _variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope)

    @property
    def scope(self):
        name = "%s_%s" % (self.__class__.__name__, self.suffix)
        return name.replace('-', '_').replace(':', '_').replace(' ', '_').replace('.', '_')

    def _initialize_variables(self):
        self.sess.run(tf.initialize_variables(self._variables))

    def predict(self, s):
        raise NotImplementedError

    def train(self, episodes, steps, on_episode):
        raise NotImplementedError

    def save_weights(self, path):
        saver = tf.train.Saver(self._variables)
        saver.save(self.sess, path)

    def restore_weights(self, path):
        saver = tf.train.Saver(self._variables)
        if not os.path.exists(path):
            raise ValueError("File not found: '%s'" % path)
        saver.restore(self.sess, path)
