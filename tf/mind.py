from __future__ import print_function

import os

from core.context import Context
from utils.string_tools import tab


class TensorflowMind:
    def __init__(self, agent, algorithm_class):
        self.world = Context.world
        self.agent = agent
        self._algorithm = None
        self._algorithm_class = algorithm_class
        self._logger = Context.logger
        self._saved_episode = None

    def __str__(self):
        return "%s:\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "saved_episode: %s" % self._saved_episode,
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
            reward = self.world.run_episode(steps)
            self._logger.on_evaluiation_end(ep, reward)

    def save(self):
        self._logger.log('Saving %s mind...' % self.agent.full_id)
        self.algorithm.save(self.data_path)

    def restore(self):
        self.algorithm.restore(self.data_path)
        self._saved_episode = self.algorithm.episode

    def try_restore(self):
        try:
            self.restore()
        except (ValueError, IOError):
            if not self.agent.is_training:
                print("Can't restore mind for '%s' from path '%s'" % (self.agent.full_id, self.data_path))
            return False
        return True

    def _save_results_if_need(self, ep, eps, sve):
        if (ep > self._saved_episode and (ep + 1) % sve == 0) or (ep == eps):
            self._saved_episode = ep
            self.save()

    @property
    def data_path(self):
        path = Context.config.get('env.%s.mind_path' % self.agent.full_id, None)
        if path is not None:
            return path
        return os.path.join(Context.work_path, 'mind', self.agent.full_id)
