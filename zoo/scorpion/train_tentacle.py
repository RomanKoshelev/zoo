from __future__ import print_function

from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from alg.dummy_alg import DummyAlgorithm
from config import config
from core.experiment import Experiment
from utils.noise_tools import linear_05_00
from zoo.scorpion.target import jpos_random_target

config['exp.id'] = "train_tentacle"

config['train.agent'] = "world.scorpion.tentacle"
config['env.world.scorpion.tentacle.algorithm'] = DDPG_PeterKovacs

config['exp.episodes'] = 30000
config['exp.steps'] = 150
config['exp.save_every_episodes'] = 200
config['alg.noise_rate_method'] = linear_05_00

config['env.target_range_xz'] = [[-.7, +.7], [+.5, +1.0]]
config['view.width'] = 1200
config['view.height'] = 800

config['env.world.scorpion.algorithm'] = DummyAlgorithm
config['env.episod_jpos_method'] = jpos_random_target

exp = Experiment(config)
exp.train()
