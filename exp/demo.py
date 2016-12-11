from __future__ import print_function

from core.context import Context
from core.ddpg_mind import DdpgMind
from core.demo_procedure import DemoProc
from core.experiment import Experiment
from core.mujoco_agent import MujocoAgent
from core.reporter import Reporter
from core.tensorflow_platform import TensorflowPlatform
from env.tentacle_world import TentacleWorld
from env.tentacle_reward import default_reward
from env.tentacle_target import random_target


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
        'env.target_range_xz': [1.1, 0.9],
        'env.target_location_method': random_target,

        'mind.evaluate_every_episodes': 10,

        'report.write_every_episodes': 5,
        'report.summary_every_episodes': 20,
        'report.diagram_mean_frame': 50,
        'report.refrech_html_every_secs': 30,

        'view.width': 1200,
        'view.height': 800,
    }

    demo_proc = DemoProc(TensorflowPlatform, TentacleWorld, MujocoAgent, DdpgMind, Reporter)
    experiment = Experiment("demo", demo_proc, init_from="007")
    experiment.start()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
