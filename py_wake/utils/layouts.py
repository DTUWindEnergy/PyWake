from py_wake import np
from scipy.spatial._qhull import ConvexHull


def square(N, distance):
    return rectangle(N, int(np.sqrt(N)), distance)


def rectangle(N, columns, distance):
    return np.array([np.ravel(v)[:N] * distance
                     for v in np.meshgrid(np.arange(columns), np.arange(np.ceil(N / columns)))])


def circular(wt_in_orbits=[1, 5, 10], outer_radius=1300):
    assert wt_in_orbits[0] <= 1
    return np.array([(r * np.cos(theta), r * np.sin(theta))
                     for n, r in zip(wt_in_orbits, np.linspace(0, outer_radius, len(wt_in_orbits)))
                     for theta in np.linspace(0, 2 * np.pi, n, endpoint=False)]).T


def farm_area(wt_x, wt_y):
    """
    Parameters
    ----------
    wt_x : array_like
        x-coordinate of wind turbines [m]
    wt_y : array_like
        y-coordinate of wind turbines [m]

    Returns
    -------
    y : float
        wind farm area [m^2]
    """

    return ConvexHull(points=np.array([wt_x, wt_y]).T).volume
