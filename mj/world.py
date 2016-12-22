from core.world import GymWorld
from mj.agent import MujocoAgent
from utils.string_tools import tab


class MujocoWorld(GymWorld, MujocoAgent):
    def __init__(self):
        MujocoAgent.__init__(self, agent_id='world', super_agent=None)
        GymWorld.__init__(self)
        self.init_agents()

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s%s" % (
            self.__class__.__name__,
            "env_id: " + self.env_id,
            "total_obs_dim: %s" % self.total_obs_dim,
            "total_act_box: %s" % self.total_act_box[0],
            "               %s" % self.total_act_box[1],
            "model_path: %s" % self.model_path,
            "env: " + tab(self._env),
            self._str_agents(),
        )

    def run_episode(self, steps):
        s = self.reset()
        reward = 0
        for t in xrange(steps):
            self.render()
            a = self.make_actions(s)
            s, r, done, _ = self.step(a)
            reward += r
        return reward

    def select_actuators(self, prefix):
        return [a for a in self._env.actuators if a['name'].startswith(prefix + '.actuator.')]
