import os

from core.procedure import Procedure


class TrainProc(Procedure):
    def __init__(self, platform_class, world_class, agent_class, mind_class, **kwargs):
        super(self.__class__, self).__init__(platform_class, world_class, agent_class, mind_class, kwargs)

    def __call__(self, ini_path, out_path):
        episodes = self.kwargs['episodes']
        steps = self.kwargs['steps']

        platform, world, mind = self._make_instances()
        w_out_path = os.path.join(out_path, "weights.ckpt")
        w_ini_path = os.path.join(ini_path, "weights.ckpt")

        with platform, world, mind:
            print(world.summary)
            if os.path.exists(w_ini_path):
                mind.restore(w_ini_path)
            mind.train(w_out_path, episodes, steps)
