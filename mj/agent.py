from __future__ import print_function
from alg.dummy_alg import DummyAlgorithm
from core.context import Context
from utils.string_tools import tab
import numpy as np
import os


class MujocoAgent:
    def __init__(self, agent_id, super_agent):
        self.agent_id = agent_id
        self._super_agent = super_agent
        self.model_path = os.path.join(Context.config['env.assets'], "%s.xml" % agent_id)
        self.mind = None
        self.obs_dim = 0
        self.act_box = None
        self.actuators = []
        self.agents = []
        for s in Context.config.get('env.%s.agents' % self.full_id, []):
            self.agents.append(MujocoAgent(s, self))

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s%s%s" % (
            self.__class__.__name__,
            "model_path: " + self.model_path,
            "obs_dim: %s" % self.obs_dim,
            "act_box: %s" % self.act_box[0],
            "         %s" % self.act_box[1],
            "mind: " + tab(self.mind),
            self._str_actuators(),
            self._str_agents(),
        )

    def train(self):
        return self.mind.train()

    def make_actions(self, state, actions=None):
        pred = self.mind.predict(state)
        if actions is None:
            actions = [None] * Context.world.total_act_dim
        assert len(self.actuators) == len(pred), "actuators=%d pred=%d" % (len(self.actuators), len(pred))
        for j, actuator in enumerate(self.actuators):
            i = actuator['index']
            assert actions[i] is None
            actions[i] = pred[j]
        for agent in self.agents:
            actions = agent.make_actions(state, actions)
        return actions

    def init_agents(self):
        self._init_actuators()
        self._init_mind()
        for a in self.agents:
            a.init_agents()

    def _init_mind(self):
        mind_class = Context.config['exp.mind_class']
        alg_class = Context.config.get('env.%s.algorithm' % self.full_id, DummyAlgorithm)
        self.mind = mind_class(agent=self, algorithm_class=alg_class)

    def _init_actuators(self):
        self.actuators = Context.world.select_actuators(self.full_id)
        ab = [[], []]
        for a in self.actuators:
            ab[0].append(a['box'][0])
            ab[1].append(a['box'][1])
        self.act_box = np.asarray(ab)

    def _str_agents(self):
        if len(self.agents) > 0:
            arr = []
            for a in self.agents:
                arr.append("\n\t%s: %s" % (a.agent_id, tab(str(a))))
            return "\n\tagents:" + tab("".join(arr))
        return ""

    def _str_actuators(self):
        if len(self.actuators) > 0:
            arr = []
            for a in self.actuators:
                arr.append("\n\t%s %s" % (a['name'], a['box']))
            return "\n\tactuators:" + tab("".join(arr))
        return ""

    def read_model(self):
        body, actuators = self._read_model()
        for sa in self.agents:
            pl = "{{%s}}" % sa.agent_id
            if pl in body:
                b, a = sa.read_model()
                body = body.replace(pl, b)
                actuators += '\n' + a
        return body, actuators

    @property
    def full_id(self):
        if self._super_agent is not None:
            return self._super_agent.full_id + '.' + self.agent_id
        else:
            return self.agent_id

    @property
    def act_dim(self):
        return len(self.actuators)

    def scale_action(self, a):
        if len(a) > 0:
            k = (a + 1.) / 2.
            a = self.act_box[0] + (self.act_box[1] - self.act_box[0]) * k  # type: np.ndarray
            return np.clip(a, self.act_box[0], self.act_box[1])
        else:
            return a

    def _read_model(self):
        with open(self.model_path, 'r') as f:
            xml = f.read().replace('{agent}', self.full_id)
            i = xml.find("<actuator>")
            if i == -1:
                body = xml
                actuators = ""
            else:
                body = xml[:i]
                actuators = xml[i:]
        return body, actuators
