from __future__ import print_function

from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from config import config
from core.experiment import Experiment
from zoo.scorpion.rewards import tentacle_reward
from zoo.scorpion.target import jpos_random_target

config['exp.id'] = "003"
config['exp.episodes'] = 30000
config['train.agent'] = "world.scorpion.tentacle"
config['env.reward_method'] = tentacle_reward
config['env.world.scorpion.tentacle.algorithm'] = DDPG_PeterKovacs
config['env.episode_jpos_method'] = jpos_random_target

exp = Experiment(config)
exp.train()
