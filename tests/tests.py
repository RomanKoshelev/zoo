from __future__ import print_function
import gym
import os
import tensorflow as tf


# =================================================================================================================

class World:
    def __init__(self):
        self.agent = None
        self.env = None
        self.state = None
        self.step = 0
        self.done = False

    def reset(self):
        self._reset()

    def _reset(self):
        raise NotImplementedError

    def proceed(self, steps=1):
        return self._proceed(steps)

    def _proceed(self, steps):
        raise NotImplementedError


class Agent:
    def __init__(self, mind):
        self.mind = mind

    def act(self, state):
        return self._act(state)

    def _act(self, state):
        return self.mind.predict(state)


class Mind:
    def __init__(self):
        pass

    def predict(self, state):
        return self._predict(state)

    def _predict(self, state):
        raise NotImplementedError


class Procedure:
    def __init__(self):
        pass

    def run(self):
        self._run()

    def _run(self):
        raise NotImplementedError


class Experiment:
    def __init__(self, proc, world, agent, mind):
        self.proc = proc
        self.world = world
        self.agent = agent
        self.mind = mind
        self.done = False

    def execute(self):
        mind = self.mind()
        agent = self.agent(mind)
        world = self.world(agent)
        proc = self.proc(world)
        proc.run()
        self.done = True

    def summary(self):
        return "Done: %s" % self.done


# =================================================================================================================

class TentacleAndApple(World):

    def __init__(self, agent):
        World.__init__(self)
        self.agent = agent
        self.env = gym.make("TentacleAndApple-v1")
        self.reset()

    def _reset(self):
        self.state = self.env.reset()
        self.step = 0
        self.done = False

    def _proceed(self, steps=1):
        for _ in xrange(steps):
            self.env.render()
            a = self.agent.act(self.state)
            self.state, _, self.done, _ = self.env.step(a)
            self.step += 1


class Tentacle(Agent):
    def __init__(self, mind):
        Agent.__init__(self, mind)


class DDPG(Mind):

    def __init__(self):
        Mind.__init__(self)
        self.algorithm = None

    def init(self, env, sess):
        # todo: refactor
        from mind.DDPG.ddpg import DDPG_PeterKovacs
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        obs_box = [env.observation_space.low, env.observation_space.high]
        act_box = [env.action_space.low, env.action_space.high]

        path = "./weigths"

        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

        self.algorithm = DDPG_PeterKovacs(sess, env.spec.id, obs_dim, obs_box, act_dim, act_box, path)

    def _predict(self, state):
        return self.algorithm.act(state)[0]


class Play(Procedure):
    def __init__(self, world):

        Procedure.__init__(self)
        self.world = world

    def _run(self):
        print("Playing the world [%s]..." % self.world)
        self.world.play()
        print("Done")


# =================================================================================================================

def test_experiment_hierarchy():
    exp = Experiment(proc=Play, world=TentacleAndApple, agent=Tentacle, mind=DDPG)
    exp.execute()
    assert exp.done
    print("Summary: ", exp.summary())


def test_gym_env_tentacle_and_apple():
    env = gym.make("TentacleAndApple-v1")
    env.reset()
    step = 0
    a = env.action_space.sample()
    while True:
        s, r, done, info = env.step(a)
        a = env.action_space.sample()
        step += 1
        if done or step > 3000:
            break


def test_play_world_tentacle_and_apple():
    mind = DDPG()
    agent = Tentacle(mind)
    world = TentacleAndApple(agent)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    mind.init(world.env, sess)

    while not world.done:
        world.proceed()


def test_train_world_tentacle_and_apple():
    pass

# =================================================================================================================

if __name__ == '__main__':
    test_play_world_tentacle_and_apple()
