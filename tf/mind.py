from __future__ import print_function

import os

from core.context import Context
from utils.string_tools import tab
from utils.os_tools import make_dir_if_not_exists
import pickle

NETWORK_WEIGHTS_PATH = "network/weights.ckpt"
ALGORITHM_STATE_PATH = "algorithm/state.pickle"


class TensorflowMind:
    def __init__(self, agent, algorithm_class):
        self.world = Context.world
        self.agent = agent
        self._algorithm = None
        self._algorithm_class = algorithm_class
        self._logger = Context.logger
        self._saved_episode = None

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "algorithm: " + tab(self.algorithm),
        )

    @property
    def algorithm(self):
        if self._algorithm is None:
            self._algorithm = self._algorithm_class(
                session=Context.platform.session,
                scope=self.agent.full_id,
                obs_dim=self.agent.alg_obs_dim,
                act_dim=self.agent.alg_act_dim,
            )
        return self._algorithm

    def predict(self, state):
        a = self.algorithm.predict(state)
        return self.agent.scale_action(a)

    def train(self):
        def do_episode_beg():
            self.world.reset()
            return self.agent.provide_alg_obs()

        def do_episode_end(ep, reward, nr, maxq):
            cf = Context.config
            self._logger.on_train_episode(ep, nr, reward, maxq)
            self._save_results_if_need(ep, cf['exp.episodes'], cf['exp.save_every_episodes'])
            self._evaluate_if_need(ep, cf['mind.evaluate_every_episodes'], cf['exp.steps'])

        def do_step(acts):
            agent_actions = self.agent.scale_action(acts)
            _, r, done, _ = self.world.step_agent(self.agent, agent_actions)
            s = self.agent.provide_alg_obs()
            self.world.render()
            return s, r, done

        return self.algorithm.train(
            episodes=Context.config['exp.episodes'],
            steps=Context.config['exp.steps'],
            on_episode_beg=do_episode_beg,
            on_episode_end=do_episode_end,
            on_step=do_step,
        )

    def _evaluate_if_need(self, ep, evs, steps):
        if (ep + 1) % evs == 0:
            self._logger.on_evaluiation_start()
            reward = self._run_episode(steps)
            self._logger.on_evaluiation_end(ep, reward)

    def _run_episode(self, steps):
        self.world.reset()
        s = self.agent.provide_alg_obs()
        reward = 0
        for t in xrange(steps):
            self.world.render()
            a = self.predict(s)
            _, r, done, _ = self.world.step(a)
            s = self.agent.provide_alg_obs()
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
        self.algorithm.restore_weights(self.network_weights_path)

    def save_weights(self):
        self.algorithm.save_weights(make_dir_if_not_exists(self.network_weights_path))

    def save_algorithm_state(self):
        with open(make_dir_if_not_exists(self.algorithm_state_path), 'w') as f:
            pickle.dump([
                self.algorithm.episode,
                self.algorithm.buffer,
            ], f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_algorithm_state(self):
        with open(self.algorithm_state_path, 'r') as f:
            [
                self.algorithm.episode,
                self.algorithm.buffer,
            ] = pickle.load(f)
        self._saved_episode = self.algorithm.episode

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
