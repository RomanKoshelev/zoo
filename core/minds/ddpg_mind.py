import os
from core.mind import Mind
from ext.alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs


class DdpgMind(Mind):
    def __init__(self):
        Mind.__init__(self)

    def _train(self, platform, world, episodes, steps):

        # todo: refactore -- use config arg .save_every_steps
        save_every_episodes = 100

        def callback(ep):
            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                print("")
                algorithm.save(model_path)
                print("")

        algorithm = DDPG_PeterKovacs(
            platform.session,
            world.id,
            world.obs_dim,
            world.act_dim,
            world.act_box)

        # todo: refactore -- use config arg
        path = os.path.join("../out/", algorithm.scope)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

        # todo: refactore -- use the world default value
        if steps is None:
            steps = 3000

        model_path = os.path.join(path, "weights.ckpt")

        if os.path.exists(model_path):
            algorithm.restore(model_path)

        return algorithm.train(world.env, episodes, steps, callback)
