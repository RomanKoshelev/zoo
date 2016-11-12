class Mind:
    def __init__(self):
        pass

    def predict(self, state):
        return self._predict(state)

    def _predict(self, state):
        raise NotImplementedError

    def train(self, episodes, steps):
        return self._train(episodes, steps)

    def _train(self, episodes, steps):
        raise NotImplementedError
