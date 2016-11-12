from core.agent import Agent


class TentacleAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

    def _train(self, episodes, steps):
        self.mind.train(episodes, steps)
