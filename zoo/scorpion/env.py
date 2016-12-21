from __future__ import print_function

from core.context import Context
from core.mujoco_env import ZooMujocoEnv


class ScorpionEnv(ZooMujocoEnv):
    def __init__(self):
        self._resets_num = 0
        self._target_pos = None
        ZooMujocoEnv.__init__(self)

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        r = Context.config['env.reward_method'](self)

        done = False

        self._update_qpos_qvel()
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

    def _set_target(self, tpos, qpos, qvel):
        for i, j in enumerate(["target.coords.x", "target.coords.z"]):
            qposadr, qveladr, _ = self.model.joint_adr(j)
            qpos[qposadr] = tpos[i]
            qvel[qveladr] = 0.
        return qpos, qvel

    def _update_qpos_qvel(self):
        if self._target_pos is not None:
            qpos = self.model.data.qpos.ravel().copy()
            qvel = self.model.data.qvel.ravel().copy()

            tpos = self._target_pos

            qpos, qvel = self._set_target(tpos, qpos, qvel)
            self.set_state(qpos, qvel)
