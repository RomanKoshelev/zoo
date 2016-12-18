from __future__ import print_function

from alg.random_alg import RandomAlgorithm
from config import config
from core.experiment import Experiment

config['exp.id'] = "002"
config['env.target_range_xz'] = [1.0, 0.5]
config['view.width'] = 1200
config['view.height'] = 800

exp = Experiment(config)
exp.demo()
