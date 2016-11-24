def staircase_5(progress):
    assert 0. <= progress <= 1.

    nr = 1.

    if progress > .01:
        nr = 0.7
    if progress > .05:
        nr = 0.5
    if progress > .4:
        nr = 0.3
    if progress > .6:
        nr = 0.1
    if progress > .9:
        nr = 0.0

    return nr


def linear_2(progress):
    assert 0. <= progress <= 1.

    return 2 * (1 - progress)
