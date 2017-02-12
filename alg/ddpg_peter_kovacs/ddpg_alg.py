import numpy as np
import tensorflow as tf

from alg.ddpg_peter_kovacs import config as cfg
from .actor import ActorNetwork
from tf.algorithm import TensorflowAlgorithm
from .buffer import ReplayBuffer
from .critic import CriticNetwork
from core.context import Context
from utils.ou_noise import OUNoise
from utils.string_tools import tab


class DDPG_PeterKovacs(TensorflowAlgorithm):
    def __init__(self, session, scope, obs_dim, act_dim):
        TensorflowAlgorithm.__init__(self, session, scope, obs_dim, act_dim)
        self.buffer = None
        with tf.variable_scope(self.scope):
            with tf.variable_scope('actor'):
                self.actor = ActorNetwork(session, obs_dim, act_dim, cfg)
            with tf.variable_scope('critic'):
                self.critic = CriticNetwork(session, obs_dim, act_dim, cfg)
        self._initialize_variables()

    def __str__(self):
        return "%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "scope: %s" % self.scope,
            "obs_dim: %d" % self._obs_dim,
            "act_dim: %d" % self._act_dim,
            "buffer: " + tab(self.buffer),
            "episode: " + str(self.episode),
        )

    def predict(self, s):
        return self.actor.predict([s])[0]

    def train(self, episodes, steps, on_episode_beg, on_episode_end, on_step):
        first_episide = self.episode + 1 if self.episode is not None else 0
        expl = self._create_exploration()

        if self.buffer is None:
            self.buffer = self._create_buffer()

        for self.episode in range(first_episide, episodes + 1):
            s = on_episode_beg()

            nrate = self._get_noise_rate(self.episode, episodes)
            reward = 0
            qmax = []

            if self.episode % 100 == 0:
                expl.reset()

            for _ in range(steps):
                # play
                a = self._make_noisy_action(s, expl.noise(), nrate)
                s2, r, done = on_step(a)
                self.buffer.add(s, a, r, s2, done)
                s = s2

                # learn
                bs, ba, br, bs2, bd = self._get_batch()
                y = self._make_target(br, bs2, bd)
                q = self._update_critic(y, bs, ba)
                self._update_actor(bs)
                self._update_target_networks()

                # statistic
                reward += r
                qmax.append(q)

                if done:
                    break

            on_episode_end(self.episode, reward, nrate, np.mean(qmax))

    def _make_noisy_action(self, s, noise, noise_rate):
        def add_noise(a, n, k):
            return (1 - k) * a + k * n
        act = self.actor.predict([s])[0]
        act = add_noise(act, noise, noise_rate)
        return act

    def _make_target(self, r, s2, done):
        q = self.critic.target_predict(s2, self.actor.target_predict(s2))
        y = []
        for i in range(len(s2)):
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
        return OUNoise(self._act_dim, mu=0,
                       sigma=Context.config['alg.noise_sigma'],
                       theta=Context.config['alg.noise_theta'])

    @staticmethod
    def _get_noise_rate(episode, episodes):
        return Context.config['alg.noise_rate_method'](episode / float(episodes))

    @staticmethod
    def _create_buffer():
        return ReplayBuffer(Context.config['alg.buffer_size'])

    def _save_state(self, path):
        self._do_save_state([self.episode, self.buffer], path)

    def _restore_state(self, path):
        [self.episode, self.buffer] = self._do_restore_state(path)
