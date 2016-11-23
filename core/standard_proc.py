class StandardProc:
    def __init__(self, platform_class, world_class, agent_class, mind_class):
        self.platform_class = platform_class
        self.world_class = world_class
        self.agent_class = agent_class
        self.mind_class = mind_class

    def train(self, weigts_path, episodes, steps):
        platform, world, mind = self._make_instances()
        with platform, world, mind:
            print(world.summary)
            mind.restore(weigts_path)
            mind.train(weigts_path, episodes, steps)

    def demo(self, weigts_path, episodes, steps):
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

    def _make_instances(self):
        platform = self.platform_class()
        agent = self.agent_class()
        world = self.world_class(agent)
        mind = self.mind_class(platform, world)
        return platform, world, mind

