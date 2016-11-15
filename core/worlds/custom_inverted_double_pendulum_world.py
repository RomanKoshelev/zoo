from core.worlds.gym_world import GymWorld


class CustomInvertedDoublePendulumWorld(GymWorld):

    def __init__(self, agent=None):
        GymWorld.__init__(self, "CustomInvertedDoublePendulum-v1", agent)
