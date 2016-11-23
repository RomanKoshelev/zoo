from ext.alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs


class DdpgMind:
    def __init__(self, platform, world):
        self.platform = platform
        self.world = world
        self._algorithm = None

    def __enter__(self):
        self._algorithm = DDPG_PeterKovacs(
            self.platform.session,
            self.world.id,
            self.world.obs_dim,
            self.world.act_dim,
            self.world.act_box)
        return self

    def __exit__(self, *args):
        pass

    def restore(self, path):
        self._algorithm.restore(path)

    def save(self, path):
        self._algorithm.save(path)

    def predict(self, state):
        a = self._algorithm.act(state)
        return self._algorithm.world_action(a)

    def train(self, weigts_path, episodes, steps):

        def callback(ep):
            save_every_episodes = 100
            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                print("")
                self._algorithm.save(weigts_path)
                print("")

        return self._algorithm.train(self.world._env, episodes, steps, callback)
