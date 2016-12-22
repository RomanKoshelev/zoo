from alg.dummy_alg import DummyAlgorithm
from core.context import Context
from utils.string_tools import tab
import os


class MujocoAgent:
    def __init__(self, agent_id, super_agent):
        self.agent_id = agent_id
        self._super_agent = super_agent
        self.model_path = os.path.join(Context.config['env.assets'], "%s.xml" % agent_id)
        self.mind = None
        self.actuators = []
        self.agents = []
        for s in Context.config.get('env.%s.agents' % self.full_id, []):
            self.agents.append(MujocoAgent(s, self))

    def __str__(self):
        return "%s:\n\t%s\n\t%s%s%s" % (
            self.__class__.__name__,
            "model_path: " + self.model_path,
            "mind: " + tab(self.mind),
            self._str_actuators(),
            self._str_agents(),
        )

    def train(self):
        return self.mind.train()

    def predict(self, state):
        return self.mind.predict(state)

    def init_agents(self):
        self._init_mind()
        self._init_actuators()
        for a in self.agents:
            a.init_agents()

    def _init_mind(self):
        mind_class = Context.config['exp.mind_class']
        alg_class = Context.config.get('env.%s.algorithm' % self.full_id, DummyAlgorithm)
        self.mind = mind_class(alg_class)

    def _init_actuators(self):
        self.actuators = Context.world.select_actuators(self.agent_id)

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

    def _read_model(self):
        with open(self.model_path, 'r') as f:
            xml = f.read()
            i = xml.find("<actuator>")
            if i == -1:
                return xml, ""
            else:
                return xml[:i], xml[i:]  # body, actuators
