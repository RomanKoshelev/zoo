import numpy as np
from gym import utils

from core.context import Context
from ext.gym.base.zoo_mujoco_env import ZooMujocoEnv


class TentacleEnv(ZooMujocoEnv, utils.EzPickle):
    def __init__(self):
        self.agent = Context.world.agent
        self.target_range = [1.5, 1.0]
        ZooMujocoEnv.__init__(self, model_path=self.agent.model_path, frame_skip=2)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        d = self.site_dist("head", "target")
        r = self.make_reward(d)

        done = False
        return ob, r, done, {}

    def _get_obs(self):
        return self.state_vector()

    def reset_model(self):
        qpos = self.model.data.qpos.ravel().copy()
        qvel = self.model.data.qvel.ravel().copy()
        qpos[-2:] = self.np_random.uniform(-1, +1, 2) * self.target_range
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
    def make_reward(target_dist):
        touch_radius = 0.05
        rw_touch = + 100. * ((touch_radius / max(target_dist, touch_radius)) ** 3)
        rw_dist = - 10. * (target_dist ** 2)
        return rw_touch + rw_dist
