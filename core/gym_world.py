import gym

from core.context import Context
import numpy as np

from utils.string_tools import tab


class GymWorld:

    def __init__(self, agent=None):
        Context.world = self
        self.agent = agent
        self.state = None
        self._env = None
        self._env_id = Context.config['env.id']

    def __str__(self):
        return "%s:\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "id: " + self.id,
            "agent: " + tab(self.agent),
        )

    def __enter__(self):
        self._env = gym.make(self._env_id)
        self.reset()
        return self

    # noinspection PyUnusedLocal
    def __exit__(self, *args):
        if self._env is not None:
            self._env.close()

    def render(self):
        self._env.render()

    def step(self, action):
        return self._env.step(action)

    @property
    def act_box(self):
        return [self._env.action_space.low, self._env.action_space.high]

    @property
    def obs_box(self):
        return [self._env.observation_space.low, self._env.observation_space.high]

    @property
    def obs_dim(self):
        return self._env.observation_space.shape[0]

    @property
    def act_dim(self):
        return self._env.action_space.shape[0]

    @property
    def id(self):
        return self._env.spec.id

    def reset(self):
        self.state = self._env.reset()
        return self.state

    def scale_action(self, a):
        k = (a + 1.) / 2.
        a = self.act_box[0] + (self.act_box[1] - self.act_box[0]) * k  # type: np.ndarray
        return np.clip(a, self.act_box[0], self.act_box[1])
