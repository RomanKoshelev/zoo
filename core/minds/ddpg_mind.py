import os
from core.mind import Mind
from ext.alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs


class DdpgMind(Mind):

    def __init__(self):
        Mind.__init__(self)
        self.algorithm = None
        self.platform = None
        self.world = None
        self.path = None

    def _init(self, platform, world):
        self.platform = platform
        self.world = world
        self.algorithm = DDPG_PeterKovacs(
            platform.session,
            world.id,
            world.obs_dim,
            world.act_dim,
            world.act_box)

    def _load(self, path):
        self.algorithm.restore(path)

    def _save(self, path):
        self.algorithm.save(path)

    def _predict(self, state):
        a = self.algorithm.act(state)
        return self.algorithm.world_action(a)

    def _train(self, episodes, steps):

        # todo: refactore -- use config arg .save_every_steps
        save_every_episodes = 100

        def callback(ep):
            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                print("")
                self.algorithm.save(self.path)
                print("")

        if os.path.exists(self.path):
            self.algorithm.restore(self.path)

        return self.algorithm.train(self.world.env, episodes, steps, callback)
