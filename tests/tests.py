from __future__ import print_function


class World:
    def __init__(self):
        pass

    def play(self):
        self._play()

    def _play(self):
        raise NotImplementedError


class Agent:
    def __init__(self):
        pass


class Mind:
    def __init__(self):
        pass


class Procedure:
    def __init__(self):
        pass

    def run(self):
        self._run()

    def _run(self):
        raise NotImplementedError


# =================================================================================================================


class TentacleAndApple(World):
    def __init__(self, agent):
        World.__init__(self)
        self.agent = agent

    def _play(self):
        print("Play with agent [%s]" % self.agent)


class Tentacle(Agent):
    def __init__(self, mind):
        Agent.__init__(self)
        self.mind = mind


class DDPG(Mind):
    def __init__(self):
        Mind.__init__(self)


class Play(Procedure):
    def __init__(self, world):
        Procedure.__init__(self)
        self.world = world

    def _run(self):
        print("Playing the world [%s]..." % self.world)
        self.world.play()
        print("Done")


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


class ExperimentInstance:
    def __init__(self):
        pass


def test_experiment_hierarchy():
    exp = Experiment(proc=Play, world=TentacleAndApple, agent=Tentacle, mind=DDPG)
    exp.execute()
    assert exp.done
    print("Summary: ", exp.summary())

if __name__ == '__main__':
    test_experiment()
