from core.context import Context
from core.procedure import Procedure


class DemoProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class, reporter_class):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class, reporter_class)

    # noinspection PyUnusedLocal
    def start(self, init_path, work_path):
        Context.mode = 'demo'
        episodes = Context.config['episodes']
        steps = Context.config['steps']

        self._make_instances(work_path)

        with self.platform, self.world, self.mind:
            print(self.world.summary)
            self.mind.restore_weights(init_path)
            print(str(self).replace('\t', "  "))
            for ep in xrange(episodes):
                s = self.world.reset()
                reward = 0
                for t in xrange(steps):
                    self.world.render()
                    a = self.mind.predict(s)
                    s, r, done, _ = self.world.step(a)
                    reward += r
                print("%3d  Reward = %+7.0f" % (ep, reward))
                self.update_title(ep, reward)

    @staticmethod
    def update_title(ep, reward):
        Context.window_title['episod'] = "|  %d: R = %+.0f" % (ep, reward)
