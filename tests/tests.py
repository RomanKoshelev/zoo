from __future__ import print_function


class TentacleAndApple:
    def __init__(self):
        pass


class Tentacle:
    def __init__(self):
        pass


class DDPG:
    def __init__(self):
        pass


class TrainProcedure:
    def __init__(self):
        pass


class Experiment:
    def __init__(self, world, avatar, mind):
        self.world = world
        self.avatar = avatar
        self.mind = mind

    def run_new(self, proc):
        self.world.add(self.avatar)

    def resume(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()


def test_experiment():
    exp = Experiment(world=TentacleAndApple, avatar=Tentacle, mind=DDPG)
    inst = exp.run_new(TrainProcedure())


if __name__ == '__main__':
    test_experiment()
