from core.worlds.openai_gym_world import OpenAiGymWorld


class InvertedDoublePendulumWorld(OpenAiGymWorld):

    def __init__(self):
        OpenAiGymWorld.__init__(self, "InvertedDoublePendulum-v1")

