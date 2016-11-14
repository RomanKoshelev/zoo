from core.worlds.openai_gym import OpenAiGym


class InvertedDoublePendulum(OpenAiGym):

    def __init__(self):
        OpenAiGym.__init__(self, "InvertedDoublePendulum-v1")

