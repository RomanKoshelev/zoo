from __future__ import print_function
from alg.dummy_alg import DummyAlgorithm
from core.context import Context
from utils.string_tools import tab
from asq.initiators import query
import numpy as np
import os

from utils.xml_tools import xml_content, xml_children_content


class MujocoAgent:
    def __init__(self, agent_id, super_agent):
        self.agent_id = agent_id
        self._super_agent = super_agent
        self.model_path = os.path.join(Context.config['env.assets'], "%s.xml" % agent_id)
        self.act_box = None
        self.sensors = []
        self.actuators = []
        self.observations = []
        self.mind = None
        self._create_agents()
        if self.is_training:
            Context.training_agent = self

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__ + (' ' * 10 + '>>>> training <<<<' if self.is_training else ''),
            "model_path: " + self.model_path,
            "alg_obs: %s" % self._str_alg_obs(),
            "alg_obs_dim: %d" % self.alg_obs_dim,
            "alg_act_dim: %d" % self.alg_act_dim,
            "sensors: " + tab(self._str_sensors()),
            "actuators: " + tab(self._str_actuators()),
            "observations: " + tab(self._str_observations()),
            "mind: " + tab(self.mind),
            "agents: " + tab(self._str_agents()),
        )

    def _create_agents(self):
        self.agents = []
        for a in Context.config.get('env.%s.agents' % self.full_id, []):
            agent_class = Context.config.get('env.%s.%s.class' % (self.full_id, a), MujocoAgent)
            self.agents.append(agent_class(a, self))

    def get_agent(self, agent_id):
        return query(self.agents).first(lambda a: a.agent_id == agent_id)

    def _str_agents(self):
        if len(self.agents) > 0:
            arr = []
            for a in self.agents:
                arr.append("\n\t%s: %s" % (a.full_id, tab(str(a))))
            return "".join(arr)
        return "\n\tno"

    def _str_alg_obs(self):
        alg_obs = self.provide_alg_obs()
        return '[' + ','.join(["%+.3g" % o for o in alg_obs]) + ']'

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
                val = ",".join(["%+.3g" % v if v is not None else "?" for v in o['get_val']()])
                arr.append("\n\t%s: %s=[%s]" % (o['type'], o['name'], val))
            return "".join(arr)
        return "\n\tno"

    def train(self):
        if self.is_training:
            self.mind.train()
        else:
            for a in self.agents:
                a.train()

    def demo(self):
        if self.is_training:
            self.mind.demo()
        else:
            for a in self.agents:
                a.demo()

    def restore(self):
        Context.logger.log("Restoring %s..." % self.full_id)
        for a in self.agents:
            a.restore()
        self.mind.restore()

    def try_restore(self):
        Context.logger.log("Trying restore %s..." % self.full_id)
        try:
            self.mind.try_restore()
            for a in self.agents:
                a.try_restore()
        except IOError:
            return False
        return True

    def predict_actions(self, actions, ignore=None):
        if self != ignore:
            pred = self.mind.predict(self.provide_alg_obs())
            assert len(self.actuators) == len(pred)
            for j, a in enumerate(self.actuators):
                actions[a['index']] = pred[j]
        for agent in self.agents:
            actions = agent.predict_actions(actions)
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
                'type': 'inputs',
                'name': self.inputs_name(inp),
                'get_val': lambda n=inp: self._super_agent.provide_inputs(n)
            })

    def get_observation(self, obs_id):
        return query(self.observations).first(lambda o: o['name'] == obs_id)

    def inputs_name(self, key):
        return '%s.inputs_%s' % (self.full_id, key)

    def sensor_name(self, key):
        return '%s.sensor_%s' % (self.full_id, key)

    def sensor_val(self, key):
        return self.get_observation(self.sensor_name(key))['get_val']()

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

    def provide_inputs(self, inputs_id):
        return [None]

    def provide_alg_obs(self):
        obs = []
        for o in self.observations:
            for s in o['get_val']():
                obs.append(s)
        return obs

    @property
    def full_id(self):
        if self._super_agent is not None:
            return self._super_agent.full_id + '.' + self.agent_id
        else:
            return self.agent_id

    @property
    def alg_act_dim(self):
        return len(self.actuators)

    @property
    def alg_obs_dim(self):
        return len(self.provide_alg_obs())

    @property
    def is_training(self):
        return self.full_id == Context.config.get('train.agent', None)

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
