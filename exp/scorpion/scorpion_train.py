from __future__ import print_function
from config import SCORPION_CONFIG
from core.experiment import Experiment
from utils.noise_tools import linear_05_00

config = SCORPION_CONFIG

config['exp.id'] = "scorpion_002"
config['exp.episodes'] = 30000
config['exp.steps'] = 30
config['exp.save_every_episodes'] = 200
config['alg.noise_rate_method'] = linear_05_00

exp = Experiment(config)
exp.train()
