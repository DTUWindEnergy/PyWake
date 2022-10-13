from py_wake import np


def square(N, distance):
    return rectangle(N, int(np.sqrt(N)), distance)


def rectangle(N, columns, distance):
    return [np.ravel(v)[:N] * distance for v in np.meshgrid(np.arange(columns), np.arange(np.ceil(N / columns)))]


def circular(wt_in_orbits=[1, 5, 10], outer_radius=1300):
    assert wt_in_orbits[0] <= 1
    return np.array([(r * np.cos(theta), r * np.sin(theta))
                     for n, r in zip(wt_in_orbits, np.linspace(0, outer_radius, len(wt_in_orbits)))
                     for theta in np.linspace(0, 2 * np.pi, n, endpoint=False)])
