class Experiment:
    def __init__(self, proc, world, agent, mind):
        self.proc = proc
        self.world = world
        self.agent = agent
        self.mind = mind
        self.done = False

    def execute(self):
        mind = self.mind()
        agent = self.agent(mind)
        world = self.world(agent)
        proc = self.proc(world)
        proc.run()
        self.done = True

    def summary(self):
        return "Done: %s" % self.done
