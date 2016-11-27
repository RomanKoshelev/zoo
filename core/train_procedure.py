from core.context import Context
from core.procedure import Procedure


class TrainProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class, reporter_class):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class, reporter_class)

    # noinspection PyUnusedLocal
    def start(self, init_path, work_path):
        Context.mode = 'train'
        self._make_instances(work_path)

        with self.platform, self.world, self.mind:
            print(self.world.summary)
            self.reporter.on_start()
            self.mind.train(work_path)

    def proceed(self, init_path, work_path):
        Context.mode = 'train'
        self._make_instances(work_path)

        with self.platform, self.world, self.mind:
            print(Context.config)
            print(self.world.summary)
            print("\nRestoring [%s] ...\n" % init_path)
            self.mind.restore(init_path)
            print(str(self).replace('\t', "  "))
            self.reporter.on_start()
            self.mind.train(work_path)
