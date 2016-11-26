from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.tensorflow_platform import TensorflowPlatform
from core.tentacle_agent import TentacleAgent
from core.tentacle_world import TentacleWorld
from core.train_procedure import TrainProc
from utils.noise_tools import staircase_5
import os


def train_mujoco_tentacle_world():
    Context.config = {
        'episodes': 100,
        'steps': 75,
        'env.model_dir': "out/tmp/",
        'world.model_path': "env/assets/tentacle_world.xml",
        'agent.model_path': "env/assets/tentacle_agent.xml",
        'mind.save_every_episodes': 10,
        'alg.batch_size': 640,
        'alg.buffer_size': 1e5,
        'alg.noise_sigma': .1,
        'alg.noise_theta': .01,
        'alg.noise_rate_method': staircase_5,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, TentacleAgent, DdpgMind)
    experiment = Experiment("001", train_proc)

    if os.path.exists(experiment.work_path):
        experiment.proceed()
    else:
        experiment.start()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
