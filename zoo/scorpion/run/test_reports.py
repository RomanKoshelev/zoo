from __future__ import print_function

from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from core.experiment import Experiment
from utils.os_tools import main_script_name
from zoo.scorpion.config import config
from zoo.scorpion.task_done import done_ball_under_1
from zoo.scorpion.task_rewards import ball_height_reward
from zoo.scorpion.task_init_jpos import jpos_init_ball

config['exp.id'] = main_script_name()
config['exp.description'] = "make reports greate again"
config['exp.steps'] = 300
config['exp.episodes'] = 15000
config['train.agent'] = "world.scorpion"
config['env.reward_method'] = ball_height_reward
config['env.world.scorpion.algorithm'] = DDPG_PeterKovacs
config['env.world.scorpion.tentacle.algorithm'] = DDPG_PeterKovacs
config['env.world.scorpion.tentacle.mind_path'] = '../experiments/003/mind/world.scorpion.tentacle'
config['env.world.scorpion.inputs'] = ['ball_x', 'ball_y', 'ball_z', 'ball_vx', 'ball_vy', 'ball_vz']
config['env.episode_jpos_method'] = jpos_init_ball
config['env.done_method'] = done_ball_under_1

exp = Experiment(config)
exp.main_report()
exp.print_instances()
