from __future__ import print_function
from config import config
from core.experiment import Experiment

config['exp.id'] = "tent_010_e30k_s50_b384"
config['exp.steps'] = 150
config['env.target_range_xz'] = [.995, .995]
config['view.width'] = 1200
config['view.height'] = 800

exp = Experiment(config)
exp.demo()
