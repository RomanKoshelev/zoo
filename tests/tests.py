from __future__ import print_function

import gym
import tensorflow as tf
from core.experiment import Experiment
from core.minds.ddpg.ddpg_mind import DdpgMind
from core.worlds.tentacle_and_apple.world import TentacleAndApple
from core.procedures.play import Play
from core.agents.tentacle import Tentacle


# =================================================================================================================

def test_experiment_hierarchy():
    exp = Experiment(proc=Play, world=TentacleAndApple, agent=Tentacle, mind=DdpgMind)
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
    mind = DdpgMind()
    agent = Tentacle(mind)
    world = TentacleAndApple(agent)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    mind.init(world.env, sess)

    while not world.done:
        world.proceed()


def test_train_world_tentacle_and_apple():
    mind = DdpgMind()
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
