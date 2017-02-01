from __future__ import print_function


# noinspection PyUnusedLocal
def zero_reward(env):
    return 0.


def tentacle_reward(env):
    target_dist = env.site_dist('world.scorpion.tentacle.site_head', 'world.scorpion.target.site_center')
    touch_radius = 0.05
    rw_touch = + 100. * ((touch_radius / max(target_dist, touch_radius)) ** 3)
    rw_dist = - 10. * (target_dist ** 2)
    return rw_touch + rw_dist


def ball_reward(env):
    # bpos = env.site_pos('world.ball')
    # hpos = env.site_pos('world.scorpion.tentacle.site_head')
    target_dist = env.site_dist('world.scorpion.tentacle.site_head', 'world.ball')
    touch_radius = 0.1
    rw_touch = + 100. * ((touch_radius / max(target_dist, touch_radius)) ** 3)
    rw_dist = - 10. * (target_dist ** 2)
    return rw_touch + rw_dist


def target_ball_reward(env):
    # bpos = env.site_pos('world.ball')
    # hpos = env.site_pos('world.scorpion.tentacle.site_head')
    target_dist = env.site_dist('world.scorpion.target.site_center', 'world.ball')
    touch_radius = 0.05
    rw_touch = + 100. * ((touch_radius / max(target_dist, touch_radius)) ** 3)
    rw_dist = - 10. * (target_dist ** 2)
    return rw_touch + rw_dist


def ball_height_reward(env):
    x, y, z = env.site_pos('world.ball')
    h = 3
    height_reward = (1 if z > h else -1) * (z - h) ** 2
    return height_reward + max(0, ball_reward(env))


def ball_very_height_reward(env):
    x, y, z = env.site_pos('world.ball')
    h = 3
    height_reward = (1 if z > h else -1) * (z - h) ** 2
    return 10 * height_reward + max(0, ball_reward(env))
