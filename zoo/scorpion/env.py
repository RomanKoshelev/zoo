from __future__ import print_function

from mujoco_py import glfw

from core.context import Context
from core.mujoco_env import ZooMujocoEnv

FRAME_SKIP = 2


class ScorpionEnv(ZooMujocoEnv):
    def __init__(self):
        self._resets_num = 0
        self._target_pos = None
        ZooMujocoEnv.__init__(self, frame_skip=FRAME_SKIP)

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        r = Context.config['env.reward_method'](self)

        done = False

        self._update_target_pos()
        self._update_window_title()
        return ob, r, done, {}

    def reset_model(self):
        self._resets_num += 1

        if Context.mode == 'train' and self._resets_num % Context.config['env.init_every_resets'] == 0:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.model.data.qpos.ravel().copy()
            qvel = self.model.data.qvel.ravel().copy()

        self.set_state(qpos, qvel)

        self._target_pos = Context.config['env.target_location_method']()

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

    def _get_obs(self):
        return self.state_vector()

    def _set_target(self, tpos, qpos, qvel):
        for i, j in enumerate(["target.x", "target.z"]):
            qposadr, qveladr, _ = self.model.joint_adr(j)
            qpos[qposadr] = tpos[i]
            qvel[qveladr] = 0.
        return qpos, qvel

    def _update_target_pos(self):
        if self._target_pos is not None:
            qpos = self.model.data.qpos.ravel().copy()
            qvel = self.model.data.qvel.ravel().copy()

            if Context.config['env.target_mouse_control']:
                tpos = self._target_from_mouse_pos()
            else:
                tpos = self._target_pos

            qpos, qvel = self._set_target(tpos, qpos, qvel)
            self.set_state(qpos, qvel)

    def _target_from_mouse_pos(self):
        mx, my = glfw.get_cursor_pos(self.viewer.window)
        w = self.viewer.get_rect().width
        h = self.viewer.get_rect().height
        x = 4.7 * (0.5 - mx / w)
        z = 3.0 * (0.4 - my / h)
        return [x, z]
