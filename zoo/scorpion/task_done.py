from __future__ import print_function


# noinspection PyUnusedLocal
def done_false(env):
    return False


def done_ball_under_1(env):
    bpos = env.site_pos('world.ball')
    return bpos[2] < 1
