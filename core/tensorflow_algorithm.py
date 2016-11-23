import tensorflow as tf


class TensorflowAlgorithm(object):
    def __init__(self, sess, suffix):
        self.sess = sess
        self.suffix = suffix

    @property
    def _variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope)

    @property
    def scope(self):
        name = "%s_%s" % (self.__class__.__name__, self.suffix)
        return name.replace('-', '_').replace(':', '_').replace(' ', '_').replace('.', '_')

    def _initialize_variables(self):
        self.sess.run(tf.initialize_variables(self._variables))

    def save(self, path):
        saver = tf.train.Saver(self._variables)
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver(self._variables)
        saver.restore(self.sess, path)
