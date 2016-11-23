import gym

from core.context import Context


class GymWorld:

    def __init__(self, env_id, agent=None):
        Context.world = self
        self.agent = agent
        self.state = None
        self._env = None
        self._env_id = env_id

    def __enter__(self):
        self._env = gym.make(self._env_id)
        self.reset()
        return self

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

    @property
    def summary(self):
        r = "\n==============================================================================\n"
        r += ("obs_dim: %d\n" % self.obs_dim)
        r += ("obs_box: %s\n" % self.obs_box[0])
        r += ("         %s\n" % self.obs_box[1])
        r += ("act_dim: %d\n" % self.act_dim)
        r += ("act_box: %s\n" % self.act_box[0])
        r += ("         %s\n" % self.act_box[1])
        r += "==============================================================================\n\n"
        return r
