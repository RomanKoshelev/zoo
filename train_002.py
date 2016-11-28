from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.mujoco_agent import MujocoAgent
from core.reporter import Reporter
from core.tensorflow_platform import TensorflowPlatform
from core.train_procedure import TrainProc
from env.tentacle_world import TentacleWorld
from env.tentacle_reward import default_reward
from env.tentacle_target import random_target
from utils.noise_tools import staircase_5
import os


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 10000,
        'steps': 75,
        'save_every_episodes': 20,

        'exp.base_path': "./out/experiments",

        'env.model_dir': "out/tmp/",
        'env.world_path': "env/assets/tentacle_world.xml",
        'env.agent_path': "env/assets/tentacle_agent.xml",
        'env.target_location_method': random_target,
        'env.reward_method': default_reward,

        'mind.evaluate_every_episodes': 10,

        'alg.batch_size': 640,
        'alg.buffer_size': 1e5,
        'alg.noise_sigma': .1,
        'alg.noise_theta': .01,
        'alg.noise_rate_method': staircase_5,

        'report.write_every_episodes': 20,
        'report.summary_every_episodes': 10,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, MujocoAgent, DdpgMind, Reporter)
    experiment = Experiment("002", train_proc)

    if os.path.exists(experiment.work_path):
        experiment.proceed()
    else:
        experiment.start()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
