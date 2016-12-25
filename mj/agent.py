from __future__ import print_function
from alg.dummy_alg import DummyAlgorithm
from core.context import Context
from utils.string_tools import tab
import numpy as np
import os

from utils.xml_tools import xml_content, xml_children_content


class MujocoAgent:
    def __init__(self, agent_id, super_agent):
        self.agent_id = agent_id
        self._super_agent = super_agent
        self.model_path = os.path.join(Context.config['env.assets'], "%s.xml" % agent_id)
        self.obs_dim = 0
        self.act_box = None
        self.sensors = []
        self.actuators = []
        self.observations = []
        self.mind = None
        self.agents = []
        for s in Context.config.get('env.%s.agents' % self.full_id, []):
            self.agents.append(MujocoAgent(s, self))

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "model_path: " + self.model_path,
            "sensors: " + tab(self._str_sensors()),
            "actuators: " + tab(self._str_actuators()),
            "observations: " + tab(self._str_observations()),
            "mind: " + tab(self.mind),
            "agents: " + tab(self._str_agents()),
        )

    def _str_agents(self):
        if len(self.agents) > 0:
            arr = []
            for a in self.agents:
                arr.append("\n\t%s: %s" % (a.full_id, tab(str(a))))
            return "".join(arr)
        return "\n\tno"

    def _str_sensors(self):
        if len(self.sensors) > 0:
            arr = []
            for s in self.sensors:
                arr.append("\n\t%s [%d]" % (s['name'], s['dim']))
            return "".join(arr)
        return "\n\tno"

    def _str_actuators(self):
        if len(self.actuators) > 0:
            arr = []
            for a in self.actuators:
                arr.append("\n\t%s [%+.5g %+.5g]" % (a['name'], a['box'][0], a['box'][1]))
            return "".join(arr)
        return "\n\tno"

    def _str_observations(self):
        if len(self.observations) > 0:
            arr = []
            for o in self.observations:
                val = ",".join(["%.5g" % v for v in o['get_val']()])
                arr.append("\n\t%s: %s=[%s]" % (o['type'], o['name'], val))
            return "".join(arr)
        return "\n\tno"

    def train(self):
        return self.mind.train()

    def do_actions(self, state, actions=None):
        pred = self.mind.predict(state)
        if actions is None:
            actions = [None] * Context.world.total_act_dim
        assert len(self.actuators) == len(pred), "actuators=%d pred=%d" % (len(self.actuators), len(pred))
        for j, actuator in enumerate(self.actuators):
            i = actuator['index']
            assert actions[i] is None
            actions[i] = pred[j]
        for agent in self.agents:
            actions = agent.do_actions(state, actions)
        return actions

    def init_agents(self):
        self._init_sensors()
        self._init_observation()
        self._init_actuators()
        self._init_mind()
        for a in self.agents:
            a.init_agents()
            self.observations += a.observations

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

    def _init_sensors(self):
        self.sensors = Context.world.select_sensors(self.full_id)

    def _init_observation(self):
        self.observations = []
        for s in self.sensors:
            name = s['name']
            self.observations.append({
                'type': 'sensor',
                'name': s['name'],
                'get_val': lambda n=name: Context.world.get_sensor_val(n)
            })

        for inp in Context.config.get("env.%s.inputs" % self.full_id, []):
            self.observations.append({
                'type': 'input',
                'name': inp,
                'get_val': lambda n=inp: self._super_agent.get_input_val(n)
            })

    def read_model(self):
        body, sensors, actuators = self._read_model()
        for sa in self.agents:
            pl = "{{%s}}" % sa.agent_id
            if pl in body:
                b, s, a = sa.read_model()
                body = body.replace(pl, b)
                sensors += '\n' + s
                actuators += '\n' + a
        return body, sensors, actuators

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def get_input_val(self, key):
        return []

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
            xml = f.read().replace('{agent}', self.full_id).replace('<body', '<_body').replace('</body', '</_body')
            body = xml_content(xml, '//agent/_body') + xml_content(xml, '//agent/worldbody')
            body = body.replace('<_body', '<body').replace('</_body', '</body')
            sensors = xml_children_content(xml, '//agent/sensor')
            actuators = xml_children_content(xml, '//agent/actuator')
        return body, sensors, actuators
