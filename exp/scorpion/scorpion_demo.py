from __future__ import print_function
from config import SCORPION_CONFIG
from core.experiment import Experiment

config = SCORPION_CONFIG

config['exp.id'] = "scorpion_001"
config['view.width'] = 1200
config['view.height'] = 800
config['env.target_range_xz'] = [1.5, 0.5]

exp = Experiment(config)
exp.demo()
