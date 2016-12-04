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


def train_mujoco_tentacle_world():
    Context.config = {
        'exp.episodes': 10000,
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
        'alg.batch_size': 512,
        'alg.noise_sigma': .1,
        'alg.noise_theta': .01,
        'alg.noise_rate_method': staircase_5,

        'report.write_every_episodes': 5,
        'report.summary_every_episodes': 20,
        'report.diagram_mean_frame': 50,

        'view.width': 1200,
        'view.height': 800,
    }

    train_proc = TrainProc(TensorflowPlatform, TentacleWorld, MujocoAgent, DdpgMind, Reporter)
    experiment = Experiment("001", train_proc)

    from_scratch = False

    if from_scratch or not experiment.can_proceed():
        experiment.start()
    else:
        experiment.proceed()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
