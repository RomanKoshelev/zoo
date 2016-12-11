def staircase_4(progress):
    assert 0. <= progress <= 1.

    nr = 1.

    if progress > .01:
        nr = 0.7
    if progress > .2:
        nr = 0.5
    if progress > .5:
        nr = 0.3
    if progress > .8:
        nr = 0.1
    return nr


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


def linear_05_00(progress):
    return _linear(progress, .5, .0)


def _linear(progress, l, r):
    assert 0. <= progress <= 1.
    return l + (r - l) * progress


# noinspection PyUnusedLocal
def const_01(progress):
    return .1


if __name__ == '__main__':
    f = linear_05_00
    states = []
    N = 100
    for i in range(N):
        states.append(f(i / float(N)))
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
