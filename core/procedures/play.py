from core.procedure import Procedure


class Play(Procedure):
    def __init__(self, world):
        Procedure.__init__(self)
        self.world = world

    def _run(self):
        print("Playing the world [%s]..." % self.world)
        self.world.play()
        print("Done")
