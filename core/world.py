class World:
    def __init__(self):
        self.agent = None
        self.env = None
        self.state = None
        self.step = 0
        self.done = False

    def reset(self):
        self._reset()

    def _reset(self):
        raise NotImplementedError

    def proceed(self, steps=1):
        return self._proceed(steps)

    def _proceed(self, steps):
        raise NotImplementedError
