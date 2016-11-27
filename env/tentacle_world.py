from core.mujoco_world import MujocoWorld


class TentacleWorld(MujocoWorld):
    def __init__(self, agent):
        MujocoWorld.__init__(self, "Zoo:Mujoco:Tentacle-v1", agent)