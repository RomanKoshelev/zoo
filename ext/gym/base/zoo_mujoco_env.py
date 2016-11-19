import os

import mujoco_py
from gym.envs.mujoco.mujoco_env import MujocoEnv


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
