from __future__ import print_function
from mj.env import ZooMujocoEnv


class ScorpionEnv(ZooMujocoEnv):
    def __init__(self):
        ZooMujocoEnv.__init__(self)
