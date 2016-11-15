from core.worlds.gym_world import GymWorld


class InvertedDoublePendulumWorld(GymWorld):

    def __init__(self, agent=None):
        GymWorld.__init__(self, "Zoo:InvertedDoublePendulum-v1", agent)
