import numpy as np

from core.context import Context


def jpos_do_nothing():
    return {}


def jpos_random_target():
    k = np.random.uniform(low=0., high=1., size=2)
    p1 = np.asarray(Context.config['env.target_range_xz'])[:, 0]
    p2 = np.asarray(Context.config['env.target_range_xz'])[:, 1]
    x, z = p1 + k * (p2 - p1)
    return {
        'world.scorpion.target.coords_x': x,
        'world.scorpion.target.coords_z': z,
    }


def jpos_init_ball():
    x, y = np.random.uniform(low=-.5, high=.5, size=2)
    return {
        'world.ball.coords_x': x,
        'world.ball.coords_y': y,
        'world.ball.coords_z': -2,
    }
