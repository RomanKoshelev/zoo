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
        'episodes': 10000,
        'steps': 75,

        'view.width': 1200,
        'view.height': 800,

        'exp.base_path': "../out/experiments",
        'env.world_path': "../env/assets/tentacle_world.xml",
        'env.agent_path': "../env/assets/tentacle_agent.xml",
        'env.target_location_method': random_target,
        'env.reward_method': default_reward,
        'env.target_mouse_control': True,
    }

    demo_proc = DemoProc(TensorflowPlatform, TentacleWorld, MujocoAgent, DdpgMind, Reporter)
    experiment = Experiment("demo", demo_proc, init_from="006")
    experiment.start()

if __name__ == '__main__':
    train_mujoco_tentacle_world()
