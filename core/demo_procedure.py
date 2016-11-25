from core.context import Context
from core.procedure import Procedure
import os


class DemoProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class)

    # noinspection PyUnusedLocal
    def start(self, init_path, work_path):
        episodes = Context.config['episodes']
        steps = Context.config['steps']

        platform, world, mind = self._make_instances()

        w_ini_path = os.path.join(init_path, "weights.ckpt")

        with platform, world, mind:
            print(world.summary)
            mind.restore(w_ini_path)
            for ep in xrange(episodes):
                s = world.reset()
                reward = 0
                for t in xrange(steps):
                    world.render()
                    a = mind.predict(s)
                    s, r, done, _ = world.step(a)
                    reward += r
                print("%3d  Reward = %+7.0f" % (ep, reward))
                self.update_title(ep, reward)

    @staticmethod
    def update_title(ep, reward):
        Context.window_title['episod'] = "|  %d: R = %+.0f" % (ep, reward)
