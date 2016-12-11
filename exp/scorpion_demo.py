from __future__ import print_function

from core.ddpg_mind import DdpgMind
from core.experiment import Experiment
from core.mujoco_agent import MujocoAgent
from core.reporter import Reporter
from core.tensorflow_platform import TensorflowPlatform
from env.tentacle_world import TentacleWorld
from env.tentacle_reward import default_reward
from env.tentacle_target import random_target


config = {
    'exp.id': "scorpion_001",
    'exp.episodes': 10000,
    'exp.steps': 75,
    'exp.save_every_episodes': 200,
    'exp.base_path': "../out/experiments",

    'exp.platform_class': TensorflowPlatform,
    'exp.world_class': TentacleWorld,
    'exp.agent_class': MujocoAgent,
    'exp.mind_class': DdpgMind,
    'exp.logger_class': Reporter,

    'env.model_world_path': "../env/assets/scorpion_world.xml",
    'env.model_agent_path': "../env/assets/scorpion_agent.xml",
    'env.reward_method': default_reward,
    'env.target_mouse_control': False,
    'env.target_range_xz': [1.0, 1.0],
    'env.target_location_method': random_target,

    'mind.evaluate_every_episodes': 10,

    'report.write_every_episodes': 5,
    'report.summary_every_episodes': 20,
    'report.diagram_mean_frame': 50,
    'report.refrech_html_every_secs': 30,

    'view.width': 1200,
    'view.height': 800,
}

exp = Experiment(config)
exp.demo()
