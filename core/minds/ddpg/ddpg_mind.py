import os
from core.mind import Mind
from core.minds.ddpg.alg.ddpg_alg import DDPG_PeterKovacs


class DdpgMind(Mind):

    def __init__(self):
        Mind.__init__(self)

    def _train(self, platform, world, episodes, steps):

        # todo: refactore as argument and saver callback
        path = "./weigths"
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

        algorithm = DDPG_PeterKovacs(
            platform.session,
            world.id,
            world.obs_dim,
            world.obs_box,
            world.act_dim,
            world.act_box,
            path)

        return algorithm.train(world.env, episodes, steps, 1)
