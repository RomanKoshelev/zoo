from __future__ import print_function
from config import config
from core.experiment import Experiment


config['exp.id'] = "tent_010_e30k_s50_b384"
config['exp.episodes'] = 30000
config['exp.steps'] = 50
config['exp.save_every_episodes'] = 1000
config['alg.batch_size'] = 384

exp = Experiment(config)
exp.train()
