# noinspection PyUnusedLocal
def zero_reward(env):
    return 0.


def tentacle_reward(env):
    target_dist = env.site_dist('world.scorpion.tentacle.site_head', 'world.scorpion.target.site_center')
    touch_radius = 0.05
    rw_touch = + 100. * ((touch_radius / max(target_dist, touch_radius)) ** 3)
    rw_dist = - 10. * (target_dist ** 2)
    return rw_touch + rw_dist
