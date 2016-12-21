from __future__ import print_function
from core.mujoco_env import ZooMujocoEnv


class ScorpionEnv(ZooMujocoEnv):
    def __init__(self):
        ZooMujocoEnv.__init__(self)
