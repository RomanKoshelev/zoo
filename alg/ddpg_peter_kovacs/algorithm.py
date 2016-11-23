from __future__ import print_function

import tensorflow as tf
import config as cfg
from buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from noise import OUNoise
import numpy as np


class DDPG_PeterKovacs:
    def __init__(self, sess, world, scope):
        self.sess = sess
        self.world = world
        self.scope = scope
        self.buff = ReplayBuffer(cfg.BUFFER_SIZE)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('actor'):
                self.actor = \
                    ActorNetwork(sess, world.obs_dim, world.act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
            with tf.variable_scope('critic'):
                self.critic = \
                    CriticNetwork(sess, world.obs_dim, world.act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)

        self.sess.run(tf.initialize_variables(self.get_var_list()))

    def save(self, path):
        saver = tf.train.Saver(self.get_var_list())
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver(self.get_var_list())
        saver.restore(self.sess, path)

    def predict(self, s):
        return self.actor.predict([s])

    def train(self, episodes, steps, callback):

        noise = OUNoise(self.world.act_dim, mu=0., sigma=.2, theta=.15)

        for episode in xrange(episodes):
            s = self.world.reset()
            reward = 0
            max_q = 0

            for step in xrange(steps):
                a = self.make_action(s, noise, episode, episodes)

                r, s, done = self.execute_step(a, s)

                batch = self.get_batch()

                y = self.make_target(batch)

                max_q += self.update_critic(batch, y)

                self.update_actor(batch)

                self.update_target_networks()

                # end episode
                self.world.render()
                reward += r
                if done or (step == steps - 1):
                    print("episode: %3d  | Reward: %+7.0f  |  Qmax: %+8.2f" %
                          (episode, reward, max_q / float(step)))
                    noise.reset()
                    break

            callback(episode)

    def make_action(self, s, exploration, ep, episodes):
        a = self.actor.predict([s])
        a = self.add_noise(a, exploration.noise(), ep, episodes)  # type: np.ndarray
        return a

    def execute_step(self, a, s):
        s2, r, terminal, _ = self.world.step(self.world.scale_action(a))
        self.buff.add(s, a[0], r, s2, terminal)
        s = s2
        return r, s, terminal

    def make_target(self, batch):
        s, a, r, s2, done = self.zip_batch(batch)

        target_q = self.critic.target_predict(s2, self.actor.target_predict(s2))

        y = []
        for i in xrange(len(batch)):
            if done[i]:
                y.append(r[i])
            else:
                y.append(r[i] + cfg.GAMMA * target_q[i])
        return np.reshape(y, (-1, 1))

    def update_critic(self, batch, y):
        s, a, r, s2, done = self.zip_batch(batch)
        q, _ = self.critic.train(y, s, a)
        return np.amax(q)

    def update_actor(self, batch):
        s, a, r, s2, done = self.zip_batch(batch)
        grads = self.critic.gradients(s, self.actor.predict(s))
        self.actor.train(s, grads)

    def update_target_networks(self):
        self.actor.target_train()
        self.critic.target_train()

    def get_batch(self):
        return self.buff.getBatch(cfg.BATCH_SIZE)

    def get_var_list(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope)

    @staticmethod
    def zip_batch(batch):
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done

    @staticmethod
    def add_noise(action, noise, episode, episodes):
        nr_max = 0.5
        nr_min = 0.4
        nr_eps = min(1000., episodes / 10.)
        nk = 1 - min(1., float(episode) / nr_eps)
        k = nr_min + nk * (nr_max - nr_min)
        return (1 - k) * action + k * noise
