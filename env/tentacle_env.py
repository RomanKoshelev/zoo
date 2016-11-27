import numpy as np
from core.context import Context
from env.zoo_mujoco_env import ZooMujocoEnv

INIT_EVERY_RESET = 10


class TentacleEnv(ZooMujocoEnv):
    def __init__(self):
        self.target_range = [1.5, 1.0]
        self._resets_num = 0
        ZooMujocoEnv.__init__(self, frame_skip=2)

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        d = self._site_dist('head', 'target')
        r = self._get_reward(d)

        done = False

        self._update_window_title()
        return ob, r, done, {}

    def _get_obs(self):
        return self.state_vector()

    def reset_model(self):
        self._resets_num += 1

        if Context.config['mode'] == 'train' and self._resets_num % INIT_EVERY_RESET == 0:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.model.data.qpos.ravel().copy()
            qvel = self.model.data.qvel.ravel().copy()

        qpos[-2:] = self._get_target_pos(self.target_range)
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

    @staticmethod
    def _get_reward(target_dist):
        touch_radius = 0.05
        rw_touch = + 100. * ((touch_radius / max(target_dist, touch_radius)) ** 3)
        rw_dist = - 10. * (target_dist ** 2)
        return rw_touch + rw_dist

    @staticmethod
    def _get_target_pos(target_range):
        tx, tz = np.random.uniform(-1, +1, 2) * target_range
        return np.array([tx, tz])
