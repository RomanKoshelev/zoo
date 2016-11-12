import os
from core.mind import Mind


class DdpgMind(Mind):

    def __init__(self):
        Mind.__init__(self)
        self.algorithm = None
        self.env = None

    def init(self, env, sess):
        # todo: refactor
        self.env = env
        from core.minds.ddpg.alg.ddpg_algorithm import DDPG_PeterKovacs
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
