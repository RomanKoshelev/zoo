from __future__ import print_function
from config import SCORPION_CONFIG
from core.experiment import Experiment
from utils.noise_tools import const_01

config = SCORPION_CONFIG

config['exp.id'] = "scorpion_001"
config['exp.episodes'] = 35000
config['alg.noise_rate_method'] = const_01

exp = Experiment(config)
exp.train()
