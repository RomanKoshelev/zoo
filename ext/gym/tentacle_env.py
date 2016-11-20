import numpy as np
from gym import utils

from core.context import Context
from ext.gym.base.zoo_mujoco_env import ZooMujocoEnv


class TentacleEnv(ZooMujocoEnv, utils.EzPickle):
    def __init__(self):
        self.agent = Context.world.agent
        ZooMujocoEnv.__init__(self, model_path=self.agent.model_path, frame_skip=1)
        utils.EzPickle.__init__(self)

    # noinspection PyUnresolvedReferences
    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        d = self.site_dist("head", "target")
        r = self.make_reward(d)
        print("%5.2f %10.2f" % (d, r))

        done = False
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],  # cart x pos
            np.sin(self.model.data.qpos[1:]),  # link angles
            np.cos(self.model.data.qpos[1:]),
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 8
        v.cam.elevation = -17
        v.cam.azimuth = -90
        v.cam.lookat[2] = v.model.stat.center[2] - 2

    @staticmethod
    def make_reward(d):
        r = 0.05
        rw_touch = + 100. * ((r / max(d, r)) ** 3)
        rw_dist = - 10. * (d ** 2)
        return rw_touch + rw_dist
