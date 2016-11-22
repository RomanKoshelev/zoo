class Mind:
    def __init__(self):
        pass

    def init(self, platform, world):
        self._init(platform, world)

    def _init(self, platform, world):
        raise NotImplementedError

    def train(self, episodes=100000, steps=None):
        return self._train(episodes, steps)

    def _train(self, episodes, steps):
        raise NotImplementedError

    def predict(self, state):
        return self._predict(state)

    def _predict(self, state):
        raise NotImplementedError

    def save(self, path):
        self._save(path)

    def _save(self, path):
        raise NotImplementedError

    def load(self, path):
        self._load(path)

    def _load(self, path):
        raise NotImplementedError

