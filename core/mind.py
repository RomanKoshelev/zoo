class Mind:
    def __init__(self):
        pass

    def train(self, platform, world, episodes=100000, steps=None):
        return self._train(platform, world, episodes, steps)

    def _train(self, platform, world, episodes, steps):
        raise NotImplementedError

    def predict(self, world, state):
        return self._predict(world, state)

    def _predict(self, world, state):
        raise NotImplementedError
