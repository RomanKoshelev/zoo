import os

from core.context import Context
from utils.string_tools import tab


class Experiment:
    def __init__(self, config):
        Context.config = config
        Context.experiment = self
        self.id = config['exp.id']
        Context.work_path = self.work_path
        self._make_instances()

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "id: " + tab(self.id),
            "work_path: " + tab(self.work_path),
            "platform: " + tab(self._platform),
            "world: " + tab(self._world),
            "agent: " + tab(self._agent),
            "mind: " + tab(self._mind),
            "logger: " + tab(self._logger),
        )

    def train(self):
        Context.mode = 'train'
        self._update_title()
        with self._platform, self._world, self._mind:
            if self._mind.can_restore():
                self._mind.restore()
                self._logger.restore()
            self._print_instances()
            self._logger.on_start()
            self._mind.train()

    def demo(self):
        Context.mode = 'demo'
        self._update_title()
        episodes = Context.config['exp.episodes']
        steps = Context.config['exp.steps']

        with self._platform, self._world, self._mind:
            self._mind.restore_weights()
            self._print_instances()
            for ep in xrange(episodes):
                reward = self._mind.run_episode(steps)
                Context.window_title['episode'] = "|  %d: R = %+.0f" % (ep, reward)
                print("%3d  Reward = %+7.0f" % (ep, reward))

    def _make_instances(self):
        self._logger = Context.config['exp.logger_class']()
        self._platform = Context.config['exp.platform_class']()
        self._agent = Context.config['exp.agent_class']()
        self._world = Context.config['exp.world_class'](self._agent)
        self._mind = Context.config['exp.mind_class'](self._platform, self._world, self._logger)

    @property
    def work_path(self):
        return os.path.join(Context.config['exp.base_path'], self.id)

    def _update_title(self):
        Context.window_title['exp'] = "| %s #%s" % (Context.mode, self.id)

    def _print_instances(self):
        print(str(self).replace('\t', '  '))
