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

    def train(self, episodes, steps, callback):

        exploration = OUNoise(self.world.act_dim, mu=0., sigma=.2, theta=.15)

        for ep in xrange(episodes):
            s = self.world.reset()
            reward = 0
            max_q = 0

            for step in xrange(steps):
                self.world.render()

                a = self.actor.predict([s])
                a = self.add_noise(exploration.noise(), a, ep, episodes)

                r, s, terminal = self.execute_step(a, s)
                reward += r

                a_batch, batch, r_batch, s2_batch, s_batch, t_batch = self.sample_batch()
                y = self.make_target(batch, r_batch, s2_batch, t_batch)
                max_q = self.update_critic(a_batch, max_q, s_batch, y)
                self.update_actor(s_batch)
                self.update_target_networks()

                # end episode
                if terminal or (step == steps - 1):
                    print("ep: %3d  | Reward: %+7.0f  |  Qmax: %+8.2f" %
                          (ep, reward, max_q / float(step)))
                    exploration.reset()
                    break

            callback(ep)

    @staticmethod
    def add_noise(noise, a, ep, episodes):
        nr_max = 0.5  # 0.7
        nr_min = 0.4
        nr_eps = min(1000., episodes / 10.)
        nk = 1 - min(1., float(ep) / nr_eps)
        nr = nr_min + nk * (nr_max - nr_min)
        a = (1 - nr) * a + nr * noise  # type: np.ndarray
        return a

    def execute_step(self, a, s):
        s2, r, terminal, _ = self.world.step(self.world.scale_action(a))
        self.buff.add(s, a[0], r, s2, terminal)
        s = s2
        return r, s, terminal

    def sample_batch(self):
        batch = self.buff.getBatch(cfg.BATCH_SIZE)
        s_batch, a_batch, r_batch, s2_batch, t_batch = zip(*batch)
        return a_batch, batch, r_batch, s2_batch, s_batch, t_batch

    def make_target(self, batch, r_batch, s2_batch, t_batch):
        target_q = self.critic.target_predict(s2_batch, self.actor.target_predict(s2_batch))
        y = []
        for i in xrange(len(batch)):
            if t_batch[i]:
                y.append(r_batch[i])
            else:
                y.append(r_batch[i] + cfg.GAMMA * target_q[i])
        y = np.reshape(y, (-1, 1))
        return y

    def update_critic(self, a_batch, max_q, s_batch, y):
        predicted_q, _ = self.critic.train(y, s_batch, a_batch)
        max_q += np.amax(predicted_q)
        return max_q

    def update_actor(self, s_batch):
        grads = self.critic.gradients(s_batch, self.actor.predict(s_batch))
        self.actor.train(s_batch, grads)

    def update_target_networks(self):
        self.actor.target_train()
        self.critic.target_train()

    def predict(self, s):
        return self.actor.predict([s])

    def get_var_list(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope)

    def save(self, path):
        saver = tf.train.Saver(self.get_var_list())
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver(self.get_var_list())
        saver.restore(self.sess, path)
