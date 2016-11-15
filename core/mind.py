class Mind:
    def __init__(self):
        pass

    def train(self, platform, world, episodes=100000, steps=None):
        return self._train(platform, world, episodes, steps)

    def _train(self, platform, world, episodes, steps):
        raise NotImplementedError
