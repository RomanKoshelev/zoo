from core.agent import Agent


class StandardPendulumAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.model_path = 'standard_pendulum.xml'
