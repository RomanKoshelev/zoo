from __future__ import print_function

import os

from core.context import Context
from utils.string_tools import tab
from utils.os_tools import make_dir_if_not_exists
import pickle

NETWORK_WEIGHTS_PATH = "network/weights.ckpt"
ALGORITHM_STATE_PATH = "algorithm/state.pickle"


class TensorflowMind:
    def __init__(self, platform, world, logger):
        self.platform = platform
        self.world = world
        self._logger = logger
        self._algorithm = None
        self._saved_episode = None

    def __enter__(self):
        self._algorithm = Context.config['mind.algorithm_class'](self.platform.session, self.world)
        return self

    def __exit__(self, *args):
        pass

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "algorithm: " + tab(self._algorithm),
        )

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self.world.scale_action(a)

    def train(self):
        def on_episod(ep, reward, nr, maxq):
            self._logger.on_train_episode(ep, nr, reward, maxq)
            self._save_results_if_need(ep,
                                       Context.config['exp.episodes'],
                                       Context.config['exp.save_every_episodes'])
            self._evaluate_if_need(ep,
                                   Context.config['mind.evaluate_every_episodes'],
                                   Context.config['exp.steps'])

        return self._algorithm.train(Context.config['exp.episodes'], Context.config['exp.steps'], on_episod)

    def _evaluate_if_need(self, ep, evs, steps):
        if (ep + 1) % evs == 0:
            self._logger.on_evaluiation_start()
            reward = self.run_episode(steps)
            self._logger.on_evaluiation_end(ep, reward)

    def run_episode(self, steps):
        s = self.world.reset()
        reward = 0
        for t in xrange(steps):
            self.world.render()
            a = self.predict(s)
            s, r, done, _ = self.world.step(a)
            reward += r
        return reward

    def save(self):
        self._logger.log('Saving weights and algorithm state...')
        self.save_weights()
        self.save_algorithm_state()

    def can_restore(self):
        return os.path.exists(self.network_weights_path)

    def restore(self):
        self._logger.log('Restoring weights and algorithm state...')
        self.restore_weights()
        self.restore_algorithm_state()

    def restore_weights(self):
        self._algorithm.restore_weights(self.network_weights_path)

    def save_weights(self):
        self._algorithm.save_weights(make_dir_if_not_exists(self.network_weights_path))

    def save_algorithm_state(self):
        with open(make_dir_if_not_exists(self.algorithm_state_path), 'w') as f:
            pickle.dump([
                self._algorithm.episode,
                self._algorithm.buffer,
            ], f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_algorithm_state(self):
        with open(self.algorithm_state_path, 'r') as f:
            [
                self._algorithm.episode,
                self._algorithm.buffer,
            ] = pickle.load(f)
        self._saved_episode = self._algorithm.episode

    def _save_results_if_need(self, ep, eps, sve):
        if (ep > self._saved_episode and (ep + 1) % sve == 0) or (ep == eps):
            self._saved_episode = ep
            self.save()

    @property
    def network_weights_path(self):
        return os.path.join(Context.work_path, NETWORK_WEIGHTS_PATH)

    @property
    def algorithm_state_path(self):
        return os.path.join(Context.work_path, ALGORITHM_STATE_PATH)
