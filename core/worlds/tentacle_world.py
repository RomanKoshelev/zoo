from core.worlds.base.gym_world import GymWorld


class TentacleWorld(GymWorld):

    def __init__(self, agent=None):
        GymWorld.__init__(self, "Zoo:Mujoco:Tentacle-v1", agent)
