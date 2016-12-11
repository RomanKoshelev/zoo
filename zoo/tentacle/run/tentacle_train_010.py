from __future__ import print_function
from config import config
from core.experiment import Experiment


config['exp.id'] = "tent_010_ep50k_st50"
config['exp.episodes'] = 50000
config['exp.steps'] = 50
config['exp.save_every_episodes'] = 1000

exp = Experiment(config)
exp.train()
