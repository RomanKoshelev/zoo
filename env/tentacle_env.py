import numpy as np

from core.context import Context
from core.mujoco_env import ZooMujocoEnv

INIT_EVERY_RESETS = 10
TARGET_RANGE_XZ = [1.5, 1.0]


class TentacleEnv(ZooMujocoEnv):
    def __init__(self):
        self._resets_num = 0
        self.target_range = TARGET_RANGE_XZ
        ZooMujocoEnv.__init__(self, frame_skip=2)

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        r = Context.config['env.reward_method'](self)

        done = False

        self._update_window_title()
        return ob, r, done, {}

    def _get_obs(self):
        return self.state_vector()

    def reset_model(self):
        self._resets_num += 1

        if Context.mode == 'train' and self._resets_num % INIT_EVERY_RESETS == 0:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.model.data.qpos.ravel().copy()
            qvel = self.model.data.qvel.ravel().copy()

        qpos[-2:] = Context.config['env.target_location_method'](self)
        qvel[-2:] = np.array([0, 0])

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _reset(self):
        self.reset_model()
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 8
        v.cam.elevation = -17
        v.cam.azimuth = -90
        v.cam.lookat[2] = v.model.stat.center[2] - 2
