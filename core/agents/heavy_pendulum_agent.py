from core.agent import Agent


class HeavyPendulumAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.model_path = 'heavy_pendulum.xml'
