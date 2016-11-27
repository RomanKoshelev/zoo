from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.mujoco_agent import MujocoAgent
from core.mujoco_world import MujocoWorld
from core.tensorflow_platform import TensorflowPlatform
from core.train_procedure import TrainProc
from env.tentacle_reward import default_reward
from env.tentacle_target import target_random_uniform
from utils.noise_tools import staircase_5
import os


def train_mujoco_tentacle_world():
    Context.config = {
        'mode': 'train',
        'episodes': 10000,
        'steps': 75,

        'view.height': 800,
        'view.width': 1200,

        'env.id': "Zoo:Mujoco:Tentacle-v1",
        'env.model_dir': "out/tmp/",
        'env.world_path': "env/assets/tentacle_world.xml",
        'env.agent_path': "env/assets/tentacle_agent.xml",
        'env.target_location_method': target_random_uniform,
        'env.reward_method': default_reward,

        'mind.save_every_episodes': 100,

        'alg.batch_size': 640,
        'alg.buffer_size': 1e5,
        'alg.noise_sigma': .1,
        'alg.noise_theta': .01,
        'alg.noise_rate_method': staircase_5,
    }

    train_proc = TrainProc(TensorflowPlatform, MujocoWorld, MujocoAgent, DdpgMind)
    experiment = Experiment("001", train_proc)

    if os.path.exists(experiment.work_path):
        experiment.proceed()
    else:
        experiment.start()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
