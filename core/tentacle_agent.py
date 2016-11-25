class TentacleAgent:
    def __init__(self):
        self.model_path = 'tentacle.xml'

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "model_path: %s" % self.model_path,
        )
