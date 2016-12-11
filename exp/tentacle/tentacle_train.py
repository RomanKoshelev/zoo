from __future__ import print_function
from config import config
from core.experiment import Experiment
from utils.noise_tools import linear_05_00


config['exp.id'] = "tentacle_001"
config['exp.episodes'] = 30000
config['exp.steps'] = 50
config['exp.save_every_episodes'] = 200
config['alg.noise_rate_method'] = linear_05_00

exp = Experiment(config)
exp.train()
