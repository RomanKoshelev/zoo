from __future__ import print_function
from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from config import config
from core.experiment import Experiment
from zoo.scorpion.rewards import tentacle_reward

config['exp.id'] = "004"
config['exp.steps'] = 50
config['env.reward_method'] = tentacle_reward
config['env.world.scorpion.algorithm'] = DDPG_PeterKovacs
config['env.world.scorpion.tentacle.algorithm'] = DDPG_PeterKovacs
config['env.world.scorpion.tentacle.mind_path'] = 'experiments/003/mind/world.scorpion.tentacle'

exp = Experiment(config)
exp.demo()
