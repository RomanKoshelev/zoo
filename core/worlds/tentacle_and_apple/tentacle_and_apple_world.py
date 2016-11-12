import gym
from core.world import World


class TentacleAndApple(World):

    def __init__(self, agent):
        World.__init__(self)
        self.agent = agent
        self.env = gym.make("TentacleAndApple-v1")
        self.reset()

    def _reset(self):
        self.state = self.env.reset()
        self.step = 0
        self.done = False

    def _proceed(self, steps=1):
        for _ in xrange(steps):
            self.env.render()
            a = self.agent.act(self.state)
            self.state, _, self.done, _ = self.env.step(a)
            self.step += 1
