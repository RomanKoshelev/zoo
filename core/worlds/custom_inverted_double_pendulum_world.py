from core.worlds.openai_gym_world import GymWorld


class CustomInvertedDoublePendulumWorld(GymWorld):

    def __init__(self):
        GymWorld.__init__(self, "CustomInvertedDoublePendulum-v1")

