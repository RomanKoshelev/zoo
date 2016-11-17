from core.proc import Proc


class StandardProc(Proc):
    def __init__(self):
        super(StandardProc, self).__init__()

    def _demo(self, platform, world, mind, episodes, steps):
        print(world.summary)

        for ep in xrange(episodes):
            s = world.reset()
            reward = 0
            for t in xrange(steps):
                world.render()
                a = mind.predict(world, s)
                s, r, done, _ = world.step(a)
                reward += r
                if done or (t == steps - 1):
                    print("%3d  Reward = %+7.0f" % (ep, reward))
                    break
