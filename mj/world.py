from core.world import GymWorld
from mj.agent import MujocoAgent
from utils.string_tools import tab


class MujocoWorld(GymWorld, MujocoAgent):
    def __init__(self):
        MujocoAgent.__init__(self, agent_id='world', super_agent=None)
        GymWorld.__init__(self)
        self.init_agents()

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "env_id: " + self.env_id,
            "model_path: %s" % self.model_path,
            "state: %s " % self.provide_state(),
            "state_dim: %s " % self.state_dim,
            "act_dim: %s" % self.act_dim,
            "total_act_dim: %s" % self.total_act_dim,
            "env: " + tab(self._env),
            "sensors: " + tab(self._str_sensors()),
            "actuators: " + tab(self._str_actuators()),
            "observations: " + tab(self._str_observations()),
            "mind: " + tab(self.mind),
            "agents: " + tab(self._str_agents()),
        )

    def run_episode(self, steps):
        s = self.reset()
        reward = 0
        for t in xrange(steps):
            self.render()
            a = self.do_actions(s)
            s, r, done, _ = self.step(a)
            reward += r
        return reward

    def select_actuators(self, prefix):
        return [a for a in self._env.actuators if a['name'].startswith(prefix + '.actuator_')]

    def select_sensors(self, prefix):
        return [s for s in self._env.sensors if s['name'].startswith(prefix + '.sensor_')]

    def get_sensor_val(self, name):
        return self._env.get_sensor_val(name)

    @property
    def total_act_dim(self):
        assert len(self._env.actuators) == self._env.action_space.shape[0]
        return len(self._env.actuators)
