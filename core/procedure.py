class Procedure(object):
    def __init__(self, platform_class, world_class, agent_class, mind_class, kwargs):
        self.platform_class = platform_class
        self.world_class = world_class
        self.agent_class = agent_class
        self.mind_class = mind_class
        self.kwargs = kwargs

    def _make_instances(self):
        platform = self.platform_class()
        agent = self.agent_class()
        world = self.world_class(agent)
        mind = self.mind_class(platform, world)
        return platform, world, mind

    @property
    def setting(self):
        return self.world_class.__name__ + "_" + self.agent_class.__name__ + "_" + self.mind_class.__name__
