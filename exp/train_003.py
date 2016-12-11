from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.mujoco_agent import MujocoAgent
from core.reporter import Logger
from core.tensorflow_platform import TensorflowPlatform
from core.train_procedure import TrainProc
from env.tentacle_world import TentacleWorld
from env.tentacle_reward import default_reward
from env.tentacle_target import random_target
from utils.noise_tools import staircase_5
import os


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 20000,
        'steps': 75,
        'save_every_episodes': 200,

        'exp.base_path': "../out/experiments",

        'env.world_path': "../env/assets/tentacle_world.xml",
        'env.agent_path': "../env/assets/tentacle_agent.xml",
        'env.target_location_method': random_target,
        'env.reward_method': default_reward,

        'alg.batch_size': 1024,
        'alg.buffer_size': 1e5,
        'alg.noise_sigma': .1,
        'alg.noise_theta': .01,
        'alg.noise_rate_method': staircase_5,

        'mind.evaluate_every_episodes': 10,
        'report.write_every_episodes': 5,
        'report.summary_every_episodes': 30,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, MujocoAgent, DdpgMind, Logger)
    experiment = Experiment("003", train_proc)

    from_scratch = False

    if from_scratch or not os.path.exists(experiment.work_path):
        experiment.start()
    else:
        experiment.proceed()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
