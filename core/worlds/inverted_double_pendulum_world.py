from core.worlds.openai_gym_world import GymWorld


class InvertedDoublePendulumWorld(GymWorld):

    def __init__(self):
        GymWorld.__init__(self, "InvertedDoublePendulum-v1")

