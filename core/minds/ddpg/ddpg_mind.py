import os
from core.mind import Mind
from core.minds.ddpg.alg.ddpg_alg import DDPG_PeterKovacs


class DdpgMind(Mind):
    def __init__(self):
        Mind.__init__(self)

    def _train(self, platform, world, episodes, steps):

        # todo: refactore -- use config arg
        path = "../out/weigths"
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

        # todo: refactore -- use config arg
        save_every_episodes = 10

        def callback(ep):
            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                algorithm.save(model_path)
                print()

        algorithm = DDPG_PeterKovacs(
            platform.session,
            world.id,
            world.obs_dim,
            world.act_dim,
            world.act_box)

        model_path = os.path.join(path, algorithm.scope + ".ckpt")

        if os.path.exists(model_path):
            algorithm.restore(model_path)

        return algorithm.train(world.env, episodes, steps, callback)
