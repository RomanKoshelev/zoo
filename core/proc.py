class Proc(object):
    def __init__(self):
        pass

    def demo(self, platform, world, mind, episodes=3000, steps=None):
        self._demo(platform, world, mind, episodes, steps)

    def _demo(self, platform, world, mind, episodes, steps):
        raise NotImplementedError
