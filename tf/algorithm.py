import tensorflow as tf
import pickle
import os

from utils.os_tools import make_dir_if_not_exists


class TensorflowAlgorithm(object):
    def __init__(self, sess, scope, obs_dim, act_dim):
        self.sess = sess
        self.suffix = scope
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self.episode = None

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "scope: %s" % self.scope,
            "obs_dim: %d" % self._obs_dim,
            "act_dim: %d" % self._act_dim,
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

    def train(self, episodes, steps, on_episode_beg, on_episode_end, on_step):
        raise NotImplementedError

    def save(self, path):
        make_dir_if_not_exists(path)
        self._save_weights(self._weights_path(path))
        self._save_state(self._state_path(path))

    def restore(self, path):
        self._restore_weights(self._weights_path(path))
        self._restore_state(self._state_path(path))

    def can_restore(self, path):
        return (os.path.exists(self._weights_path(path)) and
                os.path.exists(self._state_path(path)))

    def _save_weights(self, path):
        saver = tf.train.Saver(self._variables)
        saver.save(self.sess, path)

    def _restore_weights(self, path):
        saver = tf.train.Saver(self._variables)
        if not os.path.exists(path):
            raise ValueError("File not found: '%s'" % path)
        saver.restore(self.sess, path)

    def _save_state(self, path):
        pass

    def _restore_state(self, path):
        pass

    @staticmethod
    def _do_save_state(data_list, path):
        with open(path, 'w') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _do_restore_state(path):
        with open(path, 'r') as f:
            return pickle.load(f)

    @staticmethod
    def _weights_path(path):
        return os.path.join(path, 'network_weights.ckpt')

    @staticmethod
    def _state_path(path):
        return os.path.join(path, 'algorithm_state.pickle')
