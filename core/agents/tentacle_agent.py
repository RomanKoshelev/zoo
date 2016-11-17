from core.agent import Agent


class TentacleAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.model_path = 'tentacle.xml'
