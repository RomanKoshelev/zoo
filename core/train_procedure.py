import os

from core.procedure import Procedure


class TrainProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class, **kwargs):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class, kwargs)

    def __call__(self, **kwargs):
        path = kwargs['path']
        episodes = self.kwargs['episodes']
        steps = self.kwargs['steps']

        platform, world, mind = self._make_instances()
        weigts_path = os.path.join(path, "weights.ckpt")

        with platform, world, mind:
            print(world.summary)
            if os.path.exists(weigts_path):
                mind.restore(weigts_path)
            mind.train(weigts_path, episodes, steps)
