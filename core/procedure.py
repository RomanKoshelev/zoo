class Procedure(object):
    def __init__(self, platform_class, world_class, agent_class, mind_class):
        self.platform_class = platform_class
        self.world_class = world_class
        self.agent_class = agent_class
        self.mind_class = mind_class

    def _make_instances(self):
        platform = self.platform_class()
        agent = self.agent_class()
        world = self.world_class(agent)
        mind = self.mind_class(platform, world)
        return platform, world, mind
