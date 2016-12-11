from alg.ddpg_peter_kovacs.ddpg_alg import DDPG_PeterKovacs
from core.tensorflow_mind import TensorflowMind
from core.mujoco_agent import MujocoAgent
from core.logger import Logger
from core.mujoco_world import MujocoWorld
from core.tensorflow_platform import TensorflowPlatform
from env.tentacle.tentacle_reward import default_reward
from env.tentacle.tentacle_target import random_target
from utils.noise_tools import staircase_5

config = {
    'exp.id': None,
    'exp.episodes': 30000,
    'exp.steps': 50,
    'exp.save_every_episodes': 200,
    'exp.base_path': "../../out/experiments/tentacle/",

    'exp.platform_class': TensorflowPlatform,
    'exp.mind_class': TensorflowMind,
    'exp.world_class': MujocoWorld,
    'exp.agent_class': MujocoAgent,
    'exp.logger_class': Logger,

    'env.id': "Zoo:Mujoco:Tentacle-v1",
    'env.model_world_path': "../../env/tentacle/assets/tentacle_world.xml",
    'env.model_agent_path': "../../env/tentacle/assets/tentacle_agent.xml",
    'env.reward_method': default_reward,
    'env.target_mouse_control': False,
    'env.target_range_xz': [1.0, 1.0],
    'env.init_every_resets': 30,
    'env.target_location_method': random_target,

    'mind.evaluate_every_episodes': 10,
    'mind.algorithm_class': DDPG_PeterKovacs,

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
