from core.agent import Agent


class TentacleAgent(Agent):
    def __init__(self, mind):
        Agent.__init__(self, mind)

    def _train(self, episodes, steps):
        self.mind.train(episodes, steps)