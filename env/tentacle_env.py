from __future__ import print_function

from core.context import Context
from core.mujoco_env import ZooMujocoEnv

FRAME_SKIP = 2
TARGET_RANGE = [1.5, 0., 1.0]
INIT_EVERY_RESETS = 10


class TentacleEnv(ZooMujocoEnv):
    def __init__(self):
        self._resets_num = 0
        ZooMujocoEnv.__init__(self, frame_skip=FRAME_SKIP)

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

        qpos, qvel = self._set_target(qpos, qvel)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _reset(self):
        self.reset_model()
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = -1
        v.cam.azimuth = -90
        v.cam.distance = 5.5
        v.cam.elevation = -16

    def _set_target(self, qpos, qvel):
        tpos = Context.config['env.target_location_method'](TARGET_RANGE)
        joints = ["target_x", "target_y", "target_z"]

        for i, j in enumerate(joints):
            qposadr, qveladr, _ = self.model.joint_adr(j)
            qpos[qposadr] = tpos[i]
            qvel[qveladr] = 0.

        return qpos, qvel
