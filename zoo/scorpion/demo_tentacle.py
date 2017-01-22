from __future__ import print_function

from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from alg.random_alg import RandomAlgorithm
from config import config
from core.experiment import Experiment
from zoo.scorpion.rewards import tentacle_reward
from zoo.scorpion.target import jpos_random_target

config['exp.id'] = "003"

config['env.world.scorpion.tentacle.algorithm'] = DDPG_PeterKovacs

config['exp.steps'] = 50

config['env.target_range_xz'] = [[-.7, +.7], [+.5, +1.0]]
config['view.width'] = 1200
config['view.height'] = 800

config['env.world.scorpion.algorithm'] = RandomAlgorithm
config['env.episode_jpos_method'] = jpos_random_target

config['env.reward_method'] = tentacle_reward

exp = Experiment(config)
exp.demo()
