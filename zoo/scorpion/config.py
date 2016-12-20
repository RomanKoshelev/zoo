from alg.random_alg import RandomAlgorithm
from core.mujoco_agent import MujocoAgent
from core.tensorflow_mind import TensorflowMind
from core.logger import Logger
from core.mujoco_world import MujocoWorld
from core.tensorflow_platform import TensorflowPlatform
from zoo.scorpion.reward import default_reward
from zoo.scorpion.target import random_target
from utils.noise_tools import staircase_5

config = {
    'exp.id': None,
    'exp.episodes': 30000,
    'exp.steps': 50,
    'exp.save_every_episodes': 200,
    'exp.base_path': "experiments/",

    'exp.platform_class': TensorflowPlatform,
    'exp.mind_class': TensorflowMind,
    'exp.world_class': MujocoWorld,
    'exp.logger_class': Logger,

    'env.id': "Zoo:Mujoco:Scorpion-v1",
    'env.assets': "./assets/",
    'env.world': ['scorpion', 'ball'],
    'env.world.scorpion': ['tentacle', 'target'],

    'env.reward_method': default_reward,
    'env.target_mouse_control': False,
    'env.target_range_xz': [[.0, .0], [.0, .0]],
    'env.init_every_resets': 30,
    'env.target_location_method': random_target,

    'mind.evaluate_every_episodes': 10,
    'mind.algorithm_class': RandomAlgorithm,

    'alg.buffer_size': 100 * 1000,
    'alg.batch_size': 512,
    'alg.noise_sigma': .1,
    'alg.noise_theta': .01,
    'alg.noise_rate_method': staircase_5,

    'report.write_every_episodes': 15,
    'report.summary_every_episodes': 20,
    'report.diagram_mean_frame': 50,
    'report.refresh_html_every_secs': 30,
}
