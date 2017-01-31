from __future__ import print_function
from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from config import config
from core.experiment import Experiment
from zoo.scorpion.task_rewards import target_ball_reward
from zoo.scorpion.task_init_jpos import jpos_init_ball

config['exp.id'] = "007"
config['exp.steps'] = 75
config['train.agent'] = "world.scorpion"
config['env.reward_method'] = target_ball_reward
config['env.world.scorpion.algorithm'] = DDPG_PeterKovacs
config['env.world.scorpion.tentacle.algorithm'] = DDPG_PeterKovacs
config['env.world.scorpion.tentacle.mind_path'] = 'experiments/003/mind/world.scorpion.tentacle'
config['env.world.scorpion.inputs'] = ['ball_x', 'ball_y', 'ball_z', 'ball_vx', 'ball_vy', 'ball_vz']
config['env.episode_jpos_method'] = jpos_init_ball

exp = Experiment(config)
exp.demo()
