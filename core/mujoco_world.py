from core.context import Context

from core.gym_world import GymWorld


class MujocoWorld(GymWorld):
    def __init__(self, agent):
        GymWorld.__init__(self, agent)
        self.model_path = Context.config['env.model_world_path']

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "id: " + self.id,
            "model_path: %s" % self.model_path,
            "obs_dim: %s" % self.obs_dim,
            "act_box: %s" % self.act_box[0],
            "         %s" % self.act_box[1],
        )

    def read_model(self):
        with open(self.model_path, 'r') as f:
            return f.read()
