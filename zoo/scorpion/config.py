from alg.random_alg import RandomAlgorithm
from tf.mind import TensorflowMind
from core.logger import Logger
from tf.platform import TensorflowPlatform
from zoo.scorpion.reward import default_reward
from zoo.scorpion.scorpion import ScorpionAgent
from zoo.scorpion.target import jpos_do_noting
from utils.noise_tools import staircase_5
from zoo.scorpion.world import ScorpionWorld

config = {
    'exp.id': None,
    'exp.episodes': 30000,
    'exp.steps': 50,
    'exp.save_every_episodes': 200,
    'exp.base_path': "experiments/",

    'exp.platform_class': TensorflowPlatform,
    'exp.mind_class': TensorflowMind,
    'exp.world_class': ScorpionWorld,
    'exp.logger_class': Logger,

    'env.id': "Zoo:Mujoco:Scorpion-v1",
    'env.assets': "./assets/",
    'env.frame_skip': 2,

    # world
    'env.world.agents': ['scorpion', 'ball'],
    # scorpion
    'env.world.scorpion.class': ScorpionAgent,
    'env.world.scorpion.agents': ['tentacle', 'target'],
    'env.world.scorpion.algorithm': RandomAlgorithm,
    'env.world.scorpion.inputs': ['ball_x', 'ball_y', 'ball_z'],
    # tgentacle
    'env.world.scorpion.tentacle.inputs': ['target_x', 'target_z'],
    'env.world.scorpion.tentacle.algorithm': RandomAlgorithm,

    'env.reward_method': default_reward,
    'env.episod_jpos_method': jpos_do_noting,
    'env.target_range_xz': [[.0, .0], [.0, .0]],
    'env.init_every_episods': 10,

    'mind.evaluate_every_episodes': 10,

    'train.agent': None,

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
