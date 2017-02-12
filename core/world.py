import gym

from core.context import Context


class GymWorld:
    def __init__(self):
        Context.world = self
        self.state = None
        self._env = None
        self.env_id = Context.config['env.id']
        self._env = gym.make(self.env_id)
        self.reset()

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "id: " + self.env_id,
        )

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self):
        self._env.render()

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        self.state = self._env.reset()
        return self.state
