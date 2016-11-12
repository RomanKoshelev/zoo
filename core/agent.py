class Agent:
    def __init__(self, mind):
        self.mind = mind

    def act(self, state):
        return self._act(state)

    def _act(self, state):
        return self.mind.predict(state)

    def train(self, episodes, steps):
        return self._train(episodes, steps)

    def _train(self, episodes, steps):
        return self.mind.train(episodes, steps)
