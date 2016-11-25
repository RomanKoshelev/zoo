from core.procedure import Procedure


class TrainProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class)

    def start(self, work_path):
        platform, world, mind = self._make_instances()

        with platform, world, mind:
            print(world.summary)
            mind.train(work_path)

    def proceed(self, init_path, work_path):
        platform, world, mind = self._make_instances()

        with platform, world, mind:
            print(world.summary)
            mind.restore(init_path)
            mind.train(work_path)
