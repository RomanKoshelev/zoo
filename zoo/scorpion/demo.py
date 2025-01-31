from __future__ import print_function

from config import config
from core.experiment import Experiment

config['exp.id'] = "002"
config['env.target_range_xz'] = [[-.7, +.7], [+.5, +1.0]]
config['view.width'] = 1200
config['view.height'] = 800

exp = Experiment(config)
exp.demo()
