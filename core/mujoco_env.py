import ctypes
import os

import six
import mujoco_py
import numpy as np
from mujoco_py import glfw
from gym.envs.mujoco.mujoco_env import MujocoEnv

from core.context import Context
from utils.os_tools import make_dir_if_not_exists

FRAME_SKIP = 2


class ZooMujocoEnv(MujocoEnv):
    def __init__(self):
        self.work_path = Context.work_path
        self.world = Context.world
        self.step_num = 0
        self.episod_num = 0
        self._episod_jpos = {}
        MujocoEnv.__init__(self, model_path=self._compile_model(), frame_skip=FRAME_SKIP)
        print(self.actuator_names)

    @property
    def actuator_names(self):
        start_addr = ctypes.addressof(self.model.names.contents)
        return [ctypes.string_at(start_addr + int(inc))
                for inc in self.model.name_actuatoradr.flatten()]

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        r = Context.config['env.reward_method'](self)
        done = False

        self._update_joints(self._episod_jpos)
        self._update_joints(Context.config.get('env.step_jpos_method', lambda: {})())
        self._update_title()

        self.step_num += 1
        return ob, r, done, {}

    def reset_model(self):
        self._reset_if_need()
        self._episod_jpos = Context.config.get('env.episod_jpos_method', lambda: {})()
        self._update_joints(self._episod_jpos)
        self.episod_num += 1
        return self._get_obs()

    def _reset_if_need(self):
        if self.episod_num % Context.config['env.init_every_episods'] == 0:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.model.data.qpos.ravel().copy()
            qvel = self.model.data.qvel.ravel().copy()
        self.set_state(qpos, qvel)

    def _reset(self):
        self.reset_model()
        return self._get_obs()

    def _update_joints(self, jpos):
        qpos = self.model.data.qpos.ravel().copy()
        qvel = self.model.data.qvel.ravel().copy()

        for j, pos in jpos.iteritems():
            qposadr, qveladr, _ = self.model.joint_adr(j)
            qpos[qposadr] = pos
            qvel[qveladr] = 0.

        self.set_state(qpos, qvel)

    def site_pos(self, site_name):
        idx = self.model.site_names.index(six.b(site_name))
        return np.asarray(self.model.data.site_xpos[idx])

    def site_dist(self, site_name_1, site_name_2):
        v1 = self.site_pos(site_name_1)
        v2 = self.site_pos(site_name_2)
        return np.linalg.norm(v1 - v2)

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = -1
        v.cam.azimuth = Context.config.get('view.cam_azimuth', -45)
        v.cam.distance = Context.config.get('view.cam_distance', 7)
        v.cam.elevation = Context.config.get('view.cam_elevation', 3)

    def _get_obs(self):
        return self.state_vector()

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

    def _update_title(self):
        viewer = self._get_viewer()
        window = viewer.window
        t = Context.window_title
        d = " "
        title = t['app'] + d + t['exp'] + d + t['episode'] + d + t['step'] + d + t['info']
        glfw.set_window_title(window, title)
