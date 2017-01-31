import ctypes
import os

import six
import mujoco_py
import numpy as np
from mujoco_py import glfw
from gym.envs.mujoco.mujoco_env import MujocoEnv

from core.context import Context
from utils.os_tools import make_dir_if_not_exists
from utils.string_tools import tab


class ZooMujocoEnv(MujocoEnv):
    def __init__(self):
        self.work_path = Context.work_path
        self.world = Context.world
        self.step_num = 0
        self.episode_num = 0
        self.model_path = self._compile_model()
        MujocoEnv.__init__(self, self.model_path, frame_skip=Context.config['env.frame_skip'])

    def __str__(self):
        return "%s:\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "model_path: " + self.model_path,
            "sensors: " + tab(self._str_sensors()),
            "actuators: " + tab(self._str_actuators()),
        )

    def _str_actuators(self):
        if len(self.actuators) > 0:
            arr = []
            for a in self.actuators:
                arr.append("\n\t%s [%+.5g %+.5g]" % (a['name'], a['box'][0], a['box'][1]))
            return "".join(arr)
        return "\n\tno"

    def _str_sensors(self):
        if len(self.sensors) > 0:
            arr = []
            for s in self.sensors:
                arr.append("\n\t%s [%d]" % (s['name'], s['dim']))
            return "".join(arr)
        return "\n\tno"

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        # todo: use Task object here
        r = Context.config['env.reward_method'](self)
        done = Context.config['env.done_method'](self)
        self._update_joints(Context.config.get('env.step_jpos_method', lambda: {})())

        self._update_title()
        self.step_num += 1
        return ob, r, done, {}

    def reset_model(self):
        self._reset_if_need()
        self._update_joints(Context.config.get('env.episode_jpos_method', lambda: {})())
        self.episode_num += 1
        return self._get_obs()

    def _reset_if_need(self):
        if self.episode_num % Context.config['env.init_every_episods'] == 0:
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

    def get_sensor_val(self, sensor_name):
        dims = np.asarray(self.model.sensor_dim).flat
        data = np.asarray(self.model.data.sensordata).flat
        dim_idx = self.sensor_names.index(six.b(sensor_name))
        data_idx = np.sum(dims[:dim_idx])
        return np.asarray([data[data_idx + i] for i in xrange(dims[dim_idx])])

    def _get_sensor_dim(self, idx):
        dims = np.asarray(self.model.sensor_dim).flat
        return dims[idx]

    def site_dist(self, site_name_1, site_name_2):
        v1 = self.site_pos(site_name_1)
        v2 = self.site_pos(site_name_2)
        return np.linalg.norm(v1 - v2)

    @property
    def actuator_names(self):
        start_addr = ctypes.addressof(self.model.names.contents)
        return [ctypes.string_at(start_addr + int(inc))
                for inc in self.model.name_actuatoradr.flatten()]

    @property
    def sensor_names(self):
        start_addr = ctypes.addressof(self.model.names.contents)
        return [ctypes.string_at(start_addr + int(inc))
                for inc in self.model.name_sensoradr.flatten()]

    @property
    def actuators(self):
        arr = []
        for i, n in enumerate(self.actuator_names):
            arr.append({
                'index': i,
                'name': n,
                'box': [self.action_space.low[i], self.action_space.high[i]]
            })
        return arr

    @property
    def sensors(self):
        arr = []
        for i, n in enumerate(self.sensor_names):
            arr.append({
                'index': i,
                'name': n,
                'dim': self._get_sensor_dim(i),
            })
        return arr

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

        world, sensors, actuators = self.world.read_model()
        model = model.\
            replace('{{world}}', world).\
            replace("{{sensors}}", sensors). \
            replace("{{actuators}}", actuators)

        path = os.path.join(Context.work_path, 'environment/env_model.xml')
        path = os.path.abspath(path)
        make_dir_if_not_exists(path)

        with open(path, 'w') as f:
            f.write(model)

        return path

    def _update_title(self):
        viewer = self._get_viewer()
        window = viewer.window
        t = Context.window_title
        d = " "
        title = t['app'] + d + t['exp'] + d + t['episode'] + d + t['step'] + d + t['info']
        glfw.set_window_title(window, title)
