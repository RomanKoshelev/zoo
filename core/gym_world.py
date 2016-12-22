import gym

from core.context import Context
import numpy as np


class GymWorld:

    def __init__(self):
        Context.world = self
        self.state = None
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

    def reset(self):
        self.state = self._env.reset()
        return self.state

    def scale_action(self, a):
        if len(a) > 0:
            k = (a + 1.) / 2.
            a = self.act_box[0] + (self.act_box[1] - self.act_box[0]) * k  # type: np.ndarray
            return np.clip(a, self.act_box[0], self.act_box[1])
        else:
            return a
