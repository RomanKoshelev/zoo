from core.context import Context


class MujocoAgent:
    def __init__(self):
        self.model_path = Context.config['env.agent_path']

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "model_path: %s" % self.model_path,
        )

    def read_model(self):
        with open(self.model_path, 'r') as f:
            return f.read()
