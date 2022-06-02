from py_wake import np


def square(N, distance):
    return rectangle(N, int(np.sqrt(N)), distance)


def rectangle(N, columns, distance):
    return [np.ravel(v)[:N] * distance for v in np.meshgrid(np.arange(columns), np.arange(np.ceil(N / columns)))]
