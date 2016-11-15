from __future__ import print_function

import tensorflow as tf
import config as cfg
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise
import numpy as np


class DDPG_PeterKovacs:
    def __init__(self, sess, world_id, obs_dim, act_dim, act_box):
        self.sess = sess
        self.world_id = world_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_box = act_box
        self.buff = ReplayBuffer(cfg.BUFFER_SIZE)
        self.exploration = OUNoise(act_dim, mu=0., sigma=.2, theta=.15)

        with tf.variable_scope(self.scope):
            with tf.variable_scope("actor"):
                self.actor = ActorNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
            with tf.variable_scope("critic"):
                self.critic = CriticNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)

        var_list = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope)
        self.sess.run(tf.initialize_variables(var_list))
        self.saver = tf.train.Saver(var_list)

    def train(self, world, episodes, steps, callback):

        for ep in xrange(episodes):
            s, reward, terminal = world.reset(), 0, False
            max_q = 0

            for t in xrange(steps):
                world.render()

                # add noise to action
                nr_max = 0.7
                nr_min = 0.5
                nr_eps = min(1000., episodes / 10.)
                nk = 1 - min(1., float(ep) / nr_eps)
                nr = nr_min + nk * (nr_max - nr_min)
                a = self.actor.predict([s])
                n = self.exploration.noise()
                a = (1 - nr) * a + nr * n  # type: np.ndarray

                # execute step
                s2, r, terminal, _ = world.step(self.world_action(a))
                self.buff.add(s, a[0], r, s2, terminal)
                s = s2
                reward += r

                # sample minibatch
                batch = self.buff.getBatch(cfg.BATCH_SIZE)
                s_batch, a_batch, r_batch, s2_batch, t_batch = zip(*batch)

                # set target
                target_q = self.critic.target_predict(s2_batch, self.actor.target_predict(s2_batch))
                y = []
                for i in xrange(len(batch)):
                    if t_batch[i]:
                        y.append(r_batch[i])
                    else:
                        y.append(r_batch[i] + cfg.GAMMA * target_q[i])
                y = np.reshape(y, (-1, 1))

                # update critic
                predicted_q, _ = self.critic.train(y, s_batch, a_batch)
                max_q += np.amax(predicted_q)

                # update actor
                grads = self.critic.gradients(s_batch, self.actor.predict(s_batch))
                self.actor.train(s_batch, grads)

                # update the target networks
                self.actor.target_train()
                self.critic.target_train()

                # end episode
                if terminal or (t == steps - 1):
                    print("ep: %3d  |  NR = %.2f  |  Reward: %+7.0f  |  Qmax: %+8.2f" %
                          (ep, nr, reward, max_q / float(t)))
                    self.exploration.reset()
                    break

            callback(ep)

    def world_action(self, a):
        ak = (a + 1.) / 2.
        ae = self.act_box[0] + (self.act_box[1] - self.act_box[0]) * ak  # type: np.ndarray
        return np.clip(ae, self.act_box[0], self.act_box[1])[0]

    def run(self, env, episodes, steps):
        for ep in xrange(episodes):
            s = env.reset()
            reward = 0
            for t in xrange(steps):
                env.render()
                a = self.act(s)
                s, r, done, _ = env.step(self.world_action(a))
                reward += r
                if done or (t == steps - 1):
                    print("%3d  Reward = %+7.0f" % (ep, reward))
                    break

    def act(self, s):
        return self.actor.predict([s])

    @property
    def scope(self):
        name = "%s_%s" % (self.__class__.__name__, self.world_id)
        return name.\
            replace('-', '_').\
            replace(':', '_').\
            replace(' ', '_').\
            replace('.', '_')

    def save(self, path):
        self.saver.save(self.sess, path)
        print("Saved [%s]" % path)

    def restore(self, path):
        self.saver.restore(self.sess, path)
        print("Restored [%s]" % path)
