import os

import six
import mujoco_py
import numpy as np
from mujoco_py import glfw
from gym.envs.mujoco.mujoco_env import MujocoEnv

from core.context import Context
from utils.os_tools import make_dir_if_not_exists


# noinspection PyAbstractClass
class ZooMujocoEnv(MujocoEnv):
    def __init__(self, frame_skip):
        self.work_path = Context.work_path
        self.world = Context.world
        MujocoEnv.__init__(self, model_path=self._compile_model(), frame_skip=frame_skip)

    def site_pos(self, site_name):
        idx = self.model.site_names.index(six.b(site_name))
        return np.asarray(self.model.data.site_xpos[idx])

    def site_dist(self, site_name_1, site_name_2):
        v1 = self.site_pos(site_name_1)
        v2 = self.site_pos(site_name_2)
        return np.linalg.norm(v1 - v2)

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(
                init_height=Context.config.get('view.height', 400),
                init_width=Context.config.get('view.width', 600)
            )
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def _compile_model(self):
        with open(os.path.join(Context.config['env.assets'], 'env.xml'), 'r') as f:
            model = f.read()

        world, actuators = self.world.read_model()
        model = model.replace('{{world}}', world).replace("{{actuators}}", actuators)

        env_path = os.path.join(Context.work_path, 'environment/env_model.xml')
        env_path = os.path.abspath(env_path)
        make_dir_if_not_exists(env_path)

        with open(env_path, 'w') as f:
            f.write(model)

        return env_path

    def _update_window_title(self):
        viewer = self._get_viewer()
        window = viewer.window
        t = Context.window_title
        d = " "
        title = t['app'] + d + t['exp'] + d + t['episode'] + d + t['step'] + d + t['info']
        glfw.set_window_title(window, title)
