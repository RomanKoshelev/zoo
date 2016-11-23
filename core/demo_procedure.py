from core.procedure import Procedure


class DemoProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class)

    def run(self, weigts_path, episodes, steps):
        platform, world, mind = self._make_instances()
        with platform, world, mind:
            print(world.summary)
            mind.restore(weigts_path)
            for ep in xrange(episodes):
                s = world.reset()
                reward = 0
                for t in xrange(steps):
                    world.render()
                    a = mind.predict(s)
                    s, r, done, _ = world.step(a)
                    reward += r
                    if done or (t == steps - 1):
                        print("%3d  Reward = %+7.0f" % (ep, reward))
                        break