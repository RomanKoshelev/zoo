from core.worlds.gym_world import GymWorld


class TentacleAndAppleWorld(GymWorld):

    def __init__(self):
        GymWorld.__init__(self, "TentacleAndApple-v1")
