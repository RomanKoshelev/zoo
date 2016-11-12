from __future__ import print_function

import os

import gym
import tensorflow as tf

from core.agent import Agent
from core.experiment import Experiment
from core.mind import Mind
from core.procedure import Procedure

from worlds.tentacle_and_apple.world import TentacleAndApple

# =================================================================================================================


class Tentacle(Agent):
    def __init__(self, mind):
        Agent.__init__(self, mind)

    def _train(self, episodes, steps):
        self.mind.train(episodes, steps)


class DDPG(Mind):

    def __init__(self):
        Mind.__init__(self)
        self.algorithm = None
        self.env = None

    def init(self, env, sess):
        # todo: refactor
        self.env = env
        from minds.DDPG.ddpg import DDPG_PeterKovacs
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

    def _train(self, episodes, steps):
        print(episodes, steps)
        return self.algorithm.train(self.env, episodes, steps, 1)


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
    mind = DDPG()
    agent = Tentacle(mind)
    world = TentacleAndApple(agent)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    mind.init(world.env, sess)
    agent.train(episodes=2, steps=100)

# =================================================================================================================

if __name__ == '__main__':
    test_train_world_tentacle_and_apple()
