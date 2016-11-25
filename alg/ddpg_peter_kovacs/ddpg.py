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

        with tf.variable_scope(self.scope):
            with tf.variable_scope('actor'):
                self.actor = ActorNetwork(sess, world.obs_dim, world.act_dim, cfg)
            with tf.variable_scope('critic'):
                self.critic = CriticNetwork(sess, world.obs_dim, world.act_dim, cfg)

        self._initialize_variables()

    def predict(self, s):
        return self.actor.predict([s])

    def train(self, first_episode, last_episode, steps, on_episode):

        expl = self._create_exploration()
        buff = self._create_buffer()

        for episode in range(first_episode, last_episode):

            s = self.world.reset()
            nrate = self._get_noise_rate(episode, last_episode)
            maxq = []
            reward = 0

            if episode % 100 == 0:
                expl.reset()

            for step in xrange(steps):
                # play
                a = self._make_action(s)
                a += nrate * expl.noise()
                s2, r, done = self._world_step(a)
                buff.add(s, a, r, s2, done)
                s = s2

                # learn
                bs, ba, br, bs2, bd = self._get_batch(buff)
                y = self._make_target(br, bs2, bd)
                q = self._update_critic(y, bs, ba)
                self._update_actor(bs)
                self._update_target_networks()

                # show
                self.world.render()
                maxq.append(q)
                reward += r

                if done:
                    break

            on_episode(episode, reward, nrate, np.mean(maxq))

    def _make_action(self, s):
        return self.actor.predict([s])[0]

    def _world_step(self, a):
        s2, r, done, _ = self.world.step(self.world.scale_action(a))
        return s2, r, done

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

    @staticmethod
    def _get_batch(buff):
        batch = buff.getBatch(cfg.BATCH_SIZE)
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done

    def _create_exploration(self):
        from core.context import Context
        return OUNoise(self.world.act_dim, mu=0,
                       sigma=Context.config['train.noise_sigma'],
                       theta=Context.config['train.noise_theta'])

    @staticmethod
    def _get_noise_rate(episode, episodes):
        from core.context import Context
        return Context.config['train.noise_rate_method'](episode / float(episodes))

    @staticmethod
    def _create_buffer():
        from core.context import Context
        return ReplayBuffer(Context.config['train.buffer_size'])
