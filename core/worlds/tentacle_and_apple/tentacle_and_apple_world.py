import gym
from core.world import World


class TentacleAndAppleWorld(World):

    def __init__(self):
        World.__init__(self)
        self.env = None

    def __enter__(self):
        self.env = gym.make("TentacleAndApple-v1")
        self.reset()
        return self

    def __exit__(self, *args):
        if self.env is not None:
            self.env.close()

    def _render(self):
        pass

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

    def _get_summary(self):
        r = "\n==============================================================================\n"
        r += ("obs_dim: %d\n" % self.obs_dim)
        r += ("obs_box: %s\n" % self.obs_box[0])
        r += ("         %s\n" % self.obs_box[1])
        r += ("act_dim: %d\n" % self.act_dim)
        r += ("act_box: %s\n" % self.act_box[0])
        r += ("         %s\n" % self.act_box[1])
        r += "==============================================================================\n\n"
        return r
