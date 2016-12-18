from core.context import Context


class MujocoAgent:
    def __init__(self, key='env.model_agent_path'):
        self.model_path = Context.config[key]

    def __str__(self):
        return "%s:\n\t%s" % (
            self.__class__.__name__,
            "model_path: %s" % self.model_path,
        )

    def read_model(self):
        with open(self.model_path, 'r') as f:
            xml = f.read()
            i = xml.find("<actuator>")
            if i == -1:
                return xml, ""
            else:
                return xml[:i], xml[i:]  # body, actuator
