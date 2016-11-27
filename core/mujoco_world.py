from core.context import Context

from core.gym_world import GymWorld


class MujocoWorld(GymWorld):
    def __init__(self, env_id, agent):
        GymWorld.__init__(self, env_id, agent)
        self.model_path = Context.config['env.world_path']

    def __str__(self):
        return "%s:\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "id: " + self.id,
            "model_path: %s" % self.model_path,
        )

    def read_model(self):
        with open(self.model_path, 'r') as f:
            return f.read()
