from core.gym_world import GymWorld
from utils.string_tools import tab


class TentacleWorld(GymWorld):

    def __init__(self, agent=None):
        GymWorld.__init__(self, "Zoo:Mujoco:Tentacle-v1", agent)

    def __str__(self):
        return "%s:\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "id: " + self.id,
            "agent: " + tab(self.agent),
        )
