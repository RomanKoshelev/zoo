import gym
from core.world import World


class GymWorld(World):

    def __init__(self, env_id, agent=None):
        World.__init__(self)
        self.agent = agent
        self.env = None
        self.env_id = env_id

    def __enter__(self):
        self.env = gym.make(self.env_id)
        self.reset()
        return self

    def __exit__(self, *args):
        if self.env is not None:
            self.env.close()

    def _render(self):
        self.env.render()

    def _step(self, action):
        return self.env.step(action)

    def _get_act_box(self):
        return [self.env.action_space.low, self.env.action_space.high]

    def _get_obs_box(self):
        return [self.env.observation_space.low, self.env.observation_space.high]

    def _get_obs_dim(self):
        return self.env.observation_space.shape[0]

    def _get_act_dim(self):
        return self.env.action_space.shape[0]

    def _get_id(self):
        return self.env.spec.id

    def _reset(self):
        self.state = self.env.reset()
