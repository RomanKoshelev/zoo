from alg.dummy_alg import DummyAlgorithm
from tf.mind import TensorflowMind
from core.logger import Logger
from tf.platform import TensorflowPlatform
from zoo.scorpion.task_rewards import zero_reward
from zoo.scorpion.scorpion import ScorpionAgent
from zoo.scorpion.task_init_jpos import jpos_do_nothing
from utils.noise_tools import linear_05_00
from zoo.scorpion.world import ScorpionWorld
from zoo.scorpion.task_done import done_false

config = {
    'exp.id': None,
    'exp.group': 'scorpion',
    'exp.description': None,

    'exp.steps': 50,
    'exp.episodes': 30000,
    'exp.save_every_episodes': 100,
    'exp.base_path': "../experiments/",

    'exp.platform_class': TensorflowPlatform,
    'exp.mind_class': TensorflowMind,
    'exp.world_class': ScorpionWorld,
    'exp.logger_class': Logger,

    'env.id': "Zoo:Mujoco:Scorpion-v1",
    'env.assets': "../assets/",
    'env.frame_skip': 2,

    # world
    'env.world.agents': ['scorpion', 'ball'],
    # scorpion
    'env.world.scorpion.class': ScorpionAgent,
    'env.world.scorpion.agents': ['tentacle', 'target'],
    'env.world.scorpion.inputs': ['ball_x', 'ball_y', 'ball_z'],
    'env.world.scorpion.algorithm': DummyAlgorithm,
    # tentacle
    'env.world.scorpion.tentacle.inputs': ['target_x', 'target_z'],
    'env.world.scorpion.tentacle.algorithm': DummyAlgorithm,

    # task
    'env.reward_method': zero_reward,
    'env.episode_jpos_method': jpos_do_nothing,
    'env.step_jpos_method': jpos_do_nothing,
    'env.target_range_xz': [[-.7, +.7], [+.5, +1.0]],
    'env.init_every_episods': 30,
    'env.done_method': done_false,

    'mind.evaluate_every_episodes': 10,

    'train.agent': None,

    'alg.buffer_size': 100 * 1000,
    'alg.batch_size': 128,
    'alg.noise_sigma': .1,
    'alg.noise_theta': .01,
    'alg.noise_rate_method': linear_05_00,

    'report.write_every_episodes': 2,
    'report.summary_every_episodes': 30,
    'report.diagram_mean_frame': 50,
    'report.refresh_html_every_secs': 90,
    'report.http_root': "/rmus/",
    'report.http_home': "/rmus/experiments/scorpion/",
    'report.width': 116,

    'view.width': 800,  # 1200,
    'view.height': 600,  # 800,
}
