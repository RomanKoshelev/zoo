from core.context import Context
from core.procedure import Procedure


class DemoProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class, reporter_class):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class, reporter_class)

    # noinspection PyUnusedLocal
    def start(self, init_path, work_path):
        Context.mode = 'demo'
        episodes = Context.config['exp.episodes']
        steps = Context.config['exp.steps']

        self._make_instances(work_path)

        with self.platform, self.world, self.mind:
            self.mind.restore_weights(init_path)
            for ep in xrange(episodes):
                reward = self.mind.run_episode(steps)
                self.update_title(ep, reward)
                print("%3d  Reward = %+7.0f" % (ep, reward))

    @staticmethod
    def update_title(ep, reward):
        Context.window_title['episode'] = "|  %d: R = %+.0f" % (ep, reward)
