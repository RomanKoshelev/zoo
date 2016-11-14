from core.worlds.openai_gym import OpenAiGym


class TentacleAndApple(OpenAiGym):

    def __init__(self):
        OpenAiGym.__init__(self, "TentacleAndApple-v1")
