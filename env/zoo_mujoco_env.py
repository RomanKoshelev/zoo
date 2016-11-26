import os
import uuid

import six
import mujoco_py
from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np

from core.context import Context
from utils.os_tools import provide_dir


# noinspection PyAbstractClass
class ZooMujocoEnv(MujocoEnv):
    def __init__(self, frame_skip):
        self.world = Context.world
        self.agent = Context.world.agent
        MujocoEnv.__init__(self, model_path=self._complile_model(), frame_skip=frame_skip)

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(
                init_height=Context.config['view.height'],
                init_width=Context.config['view.width']
            )
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def _site_pos(self, site_name):
        idx = self.model.site_names.index(six.b(site_name))
        return np.asarray (self.model.data.site_xpos[idx])

    def _site_dist(self, site_name_1, site_name_2):
        v1 = self._site_pos(site_name_1)
        v2 = self._site_pos(site_name_2)
        return np.linalg.norm(v1 - v2)

    def _complile_model(self):
        world_model = self.world.read_model()
        agent_model = self.agent.read_model()
        env_model = world_model.replace('{{agent}}', agent_model)

        env_path = os.path.join(Context.config['env.model_dir'], "env_" + str(uuid.uuid4()) + ".xml")
        env_path = os.path.abspath(env_path)
        provide_dir(env_path)

        with open(env_path, 'w') as f:
            f.write(env_model)

        return env_path
