from core.ddpg_mind import DdpgMind
from core.mujoco_agent import MujocoAgent
from core.logger import Logger
from core.tensorflow_platform import TensorflowPlatform
from env.tentacle_reward import default_reward
from env.tentacle_target import random_target
from env.tentacle_world import TentacleWorld
from utils.noise_tools import staircase_5

SCORPION_CONFIG = {
    'exp.id': None,
    'exp.episodes': 30000,
    'exp.steps': 75,
    'exp.save_every_episodes': 200,
    'exp.base_path': "../../out/experiments",

    'exp.platform_class': TensorflowPlatform,
    'exp.world_class': TentacleWorld,
    'exp.agent_class': MujocoAgent,
    'exp.mind_class': DdpgMind,
    'exp.logger_class': Logger,

    'env.model_world_path': "../../env/assets/scorpion_world.xml",
    'env.model_agent_path': "../../env/assets/scorpion_agent.xml",
    'env.reward_method': default_reward,
    'env.target_mouse_control': False,
    'env.target_range_xz': [1.0, 1.0],
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
    'report.refresh_html_every_secs': 30,
}
