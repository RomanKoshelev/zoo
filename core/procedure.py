from utils.string_tools import tab


class Procedure(object):
    def __init__(self, platform_class, world_class, agent_class, mind_class, reporter_class):
        self.platform_class = platform_class
        self.world_class = world_class
        self.agent_class = agent_class
        self.mind_class = mind_class
        self.reporter_class = reporter_class

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "platform: " + tab(self.platform),
            "world: " + tab(self.world),
            "agent: " + tab(self.agent),
            "mind: " + tab(self.mind),
            "reporter: " + tab(self.reporter),
        )

    def _make_instances(self, work_path):
        self.reporter = self.reporter_class(work_path)
        self.platform = self.platform_class()
        self.agent = self.agent_class()
        self.world = self.world_class(self.agent)
        self.mind = self.mind_class(self.platform, self.world, self.reporter)
