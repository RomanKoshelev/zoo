from __future__ import print_function

import numpy as np
import tensorflow as tf

import config as cfg
from actor import ActorNetwork
from core.tensorflow_algorithm import TensorflowAlgorithm
from buffer import ReplayBuffer
from critic import CriticNetwork
from noise import OUNoise


class DDPG_PeterKovacs(TensorflowAlgorithm):
    def __init__(self, sess, world):
        super(self.__class__, self).__init__(sess, world.id)

        self.world = world
        self.buff = ReplayBuffer(cfg.BUFFER_SIZE)
        self.exploration = OUNoise(self.world.act_dim, mu=0., sigma=.2, theta=.15)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('actor'):
                self.actor = \
                    ActorNetwork(sess, world.obs_dim, world.act_dim, cfg.TAU, cfg.LRA, cfg.L2A)
            with tf.variable_scope('critic'):
                self.critic = \
                    CriticNetwork(sess, world.obs_dim, world.act_dim, cfg.TAU, cfg.LRC, cfg.L2C)

        self._initialize_variables()

    def predict(self, s):
        return self.actor.predict([s])

    def train(self, episodes, steps, on_episode, on_step):

        for episode in xrange(episodes):
            s = self.world.reset()
            self.exploration.reset()

            for step in xrange(steps):
                # play
                a = self._make_action(s)
                a = self._add_noise(a)
                r, s2, done = self._world_step(a)
                self._add_to_buffer(s, a, r, s2, done)
                s = s2

                # learn
                bs, ba, br, bs2, bd = self._get_batch()
                y = self._make_target(br, bs2, bd)
                maxq = self._update_critic(y, bs, ba)
                self._update_actor(bs)
                self._update_target_networks()

                # callback
                on_step(r, maxq)

                # end episode
                if done:
                    break

            # callback
            on_episode(episode)

    def _make_action(self, s):
        return self.actor.predict([s])[0]

    def _add_noise(self, a):
        a = a + self.exploration.noise()
        return np.clip(a, -1, 1)

    def _world_step(self, a):
        s2, r, done, _ = self.world.step(self.world.scale_action(a))
        return r, s2, done

    def _add_to_buffer(self, s, a, r, s2, done):
        self.buff.add(s, a, r, s2, done)

    def _make_target(self, r, s2, done):
        q = self.critic.target_predict(s2, self.actor.target_predict(s2))
        y = []
        for i in xrange(len(s2)):
            if done[i]:
                y.append(r[i])
            else:
                y.append(r[i] + cfg.GAMMA * q[i])
        return np.reshape(y, (-1, 1))

    def _update_critic(self, y, s, a):
        q, _ = self.critic.train(y, s, a)
        return np.amax(q)

    def _update_actor(self, s):
        grads = self.critic.gradients(s, self.actor.predict(s))
        self.actor.train(s, grads)

    def _update_target_networks(self):
        self.actor.target_train()
        self.critic.target_train()

    def _get_batch(self):
        batch = self.buff.getBatch(cfg.BATCH_SIZE)
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done
