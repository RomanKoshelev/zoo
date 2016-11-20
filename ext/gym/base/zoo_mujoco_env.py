import os

import six
import mujoco_py
from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np


class ZooMujocoEnv(MujocoEnv):
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "../assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        MujocoEnv.__init__(self, fullpath, frame_skip)

    def reset_model(self):
        raise NotImplemented

    def _step(self, action):
        raise NotImplemented

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_height=800, init_width=1200)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def site_pos(self, site_name):
        idx = self.model.site_names.index(six.b(site_name))
        return np.asarray (self.model.data.site_xpos[idx])

    def site_dist(self, site_name_1, site_name_2):
        v1 = self.site_pos(site_name_1)
        v2 = self.site_pos(site_name_2)
        return np.linalg.norm(v1 - v2)

