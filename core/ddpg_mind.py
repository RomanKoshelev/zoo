from alg.ddpg_peter_kovacs.algorithm import DDPG_PeterKovacs


class DdpgMind:
    def __init__(self, platform, world):
        self.platform = platform
        self.world = world
        self._algorithm = None

    def __enter__(self):

        def scope():
            name = "%s_%s" % (DDPG_PeterKovacs.__name__, self.world.id)
            return name. \
                replace('-', '_'). \
                replace(':', '_'). \
                replace(' ', '_'). \
                replace('.', '_')

        self._algorithm = DDPG_PeterKovacs(self.platform.session, self.world, scope())
        return self

    def __exit__(self, *args):
        pass

    def train(self, weigts_path, episodes, steps):

        def callback(ep):
            save_every_episodes = 100
            if (ep > 0 and ep % save_every_episodes == 0) or (ep == episodes - 1):
                print("")
                self._algorithm.save(weigts_path)
                print("")

        return self._algorithm.train(episodes, steps, callback)

    def save(self, path):
        self._algorithm.save(path)

    def restore(self, path):
        self._algorithm.restore(path)

    def predict(self, state):
        a = self._algorithm.predict(state)
        return self._algorithm.world_action(a)

