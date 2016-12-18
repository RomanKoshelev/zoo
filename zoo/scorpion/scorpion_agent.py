from core.mujoco_agent import MujocoAgent
from utils.string_tools import tab

TENTACLE_PL = '{{tentacle}}'


class ScorpionAgent(MujocoAgent):
    def __init__(self):
        MujocoAgent.__init__(self, 'env.model_agent_path.scorpion')
        self.tentacle = MujocoAgent('env.model_agent_path.tentacle')

    def __str__(self):
        return "%s:\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "model_path: %s" % self.model_path,
            "tentacle: %s" % tab(self.tentacle),
        )

    def read_model(self):
        b, a = MujocoAgent.read_model(self)
        if TENTACLE_PL in b:
            tb, ta = self.tentacle.read_model()
            b = b.replace(TENTACLE_PL, tb)
            a += '\n' + ta
        return b, a
