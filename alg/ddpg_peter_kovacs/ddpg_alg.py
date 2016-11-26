from __future__ import print_function

import numpy as np
import tensorflow as tf

import config as cfg
from actor import ActorNetwork
from core.tensorflow_algorithm import TensorflowAlgorithm
from buffer import ReplayBuffer
from critic import CriticNetwork
from noise import OUNoise
from core.context import Context
from utils.string_tools import tab


class DDPG_PeterKovacs(TensorflowAlgorithm):
    def __init__(self, sess, world):
        super(self.__class__, self).__init__(sess, world.id)

        self.world = world
        self.buffer = None
        self.episode = None

        with tf.variable_scope(self.scope):
            with tf.variable_scope('actor'):
                self.actor = ActorNetwork(sess, world.obs_dim, world.act_dim, cfg)
            with tf.variable_scope('critic'):
                self.critic = CriticNetwork(sess, world.obs_dim, world.act_dim, cfg)

        self._initialize_variables()

    def __str__(self):
        return "%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "world: " + tab(self.world),
            "buffer: " + tab(self.buffer),
            "episode: " + str(self.episode),
        )

    def predict(self, s):
        return self.actor.predict([s])

    def train(self, last_episode, steps, on_episode):

        first_episide = self.episode + 1 if self.episode is not None else 0

        expl = self._create_exploration()
        if self.buffer is None:
            self.buffer = self._create_buffer()

        for self.episode in range(first_episide, last_episode + 1):

            nrate = self._get_noise_rate(self.episode, last_episode)
            s = self.world.reset()
            reward = 0
            qmax = []

            if self.episode % 100 == 0:
                expl.reset()

            for step in xrange(steps):
                # play
                a = self._make_action(s)
                a = self._add_noise(a, expl.noise(), nrate)
                s2, r, done = self._world_step(a)
                self.buffer.add(s, a, r, s2, done)
                s = s2

                # learn
                bs, ba, br, bs2, bd = self._get_batch()
                y = self._make_target(br, bs2, bd)
                q = self._update_critic(y, bs, ba)
                self._update_actor(bs)
                self._update_target_networks()

                # show
                self.world.render()
                qmax.append(q)
                reward += r

                if done:
                    break

            on_episode(self.episode, reward, nrate, np.mean(qmax))

    @staticmethod
    def _add_noise(a, n, k):
        return (1 - k) * a + k * n

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

    def _get_batch(self):
        batch = self.buffer.get_batch(Context.config['alg.batch_size'])
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done

    def _create_exploration(self):
        return OUNoise(self.world.act_dim, mu=0,
                       sigma=Context.config['alg.noise_sigma'],
                       theta=Context.config['alg.noise_theta'])

    @staticmethod
    def _get_noise_rate(episode, episodes):
        return Context.config['alg.noise_rate_method'](episode / float(episodes))

    @staticmethod
    def _create_buffer():
        return ReplayBuffer(Context.config['alg.buffer_size'])
