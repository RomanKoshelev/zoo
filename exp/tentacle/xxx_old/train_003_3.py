from __future__ import print_function

from core.context import Context
from core.tensorflow_mind import TensorflowMind
from core.experiment import Experiment
from core.mujoco_agent import MujocoAgent
from core.reporter import Logger
from core.tensorflow_platform import TensorflowPlatform
from core.train_procedure import TrainProc
from env.tentacle_world import TentacleWorld
from env.tentacle_reward import default_reward
from env.tentacle_target import random_target
from utils.noise_tools import staircase_4


def train_mujoco_tentacle_world():
    Context.config = {
        'exp.episodes': 20000,
        'exp.steps': 75,
        'exp.save_every_episodes': 200,
        'exp.base_path': "../out/experiments",

        'env.model_world_path': "../env/assets/tentacle_world.xml",
        'env.model_agent_path': "../env/assets/tentacle_agent.xml",
        'env.reward_method': default_reward,
        'env.target_mouse_control': False,
        'env.target_range_xz': [1.5, 1.0],
        'env.target_location_method': random_target,

        'mind.evaluate_every_episodes': 10,

        'alg.buffer_size': 100 * 1000,
        'alg.batch_size': 1024,
        'alg.noise_sigma': .1,
        'alg.noise_theta': .01,
        'alg.noise_rate_method': staircase_4,

        'report.write_every_episodes': 5,
        'report.summary_every_episodes': 20,
        'report.diagram_mean_frame': 50,
        'report.refrech_html_every_secs': 30,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, MujocoAgent, TensorflowMind, Logger)
    experiment = Experiment("003_3", train_proc)

    from_scratch = False

    if from_scratch or not experiment.can_proceed():
        experiment.start()
    else:
        experiment.proceed()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
