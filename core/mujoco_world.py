from core.gym_world import GymWorld
from core.mujoco_agent import MujocoAgent


class MujocoWorld(GymWorld, MujocoAgent):
    def __init__(self):
        MujocoAgent.__init__(self, agent_id='world', super_agent=None)
        GymWorld.__init__(self)
        self.init_mind()

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s%s" % (
            self.__class__.__name__,
            "env_id: " + self.env_id,
            "obs_dim: %s" % self.obs_dim,
            "act_box: %s" % self.act_box[0],
            "         %s" % self.act_box[1],
            "model_path: %s" % self.model_path,
            self._str_agents(),
        )

    def run_episode(self, steps):
        s = self.reset()
        reward = 0
        for t in xrange(steps):
            self.render()
            a = self.predict(s)
            s, r, done, _ = self.step(a)
            reward += r
        return reward
