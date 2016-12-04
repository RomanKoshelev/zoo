import numpy as np


def running_mean(x, frame):
    # np.convolve(x, np.ones((w,)) / w, 'same')

    x = np.asarray(x, float)  # type: np.ndarray
    m = np.zeros(len(x))  # type: np.ndarray
    for i in range(len(x)):
        if i == 0:
            m[i] = x[i]
        else:
            m[i] = np.mean(x[max(0, i - frame):i])
    return m

