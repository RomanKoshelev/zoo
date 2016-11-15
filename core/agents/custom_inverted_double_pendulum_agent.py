from core.agent import Agent


class CustomInvertedDoublePendulumAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.model_path = 'custom_inverted_double_pendulum.xml'
